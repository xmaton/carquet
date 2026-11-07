/**
 * @file test_dictionary_codes.c
 * @brief Column readers reading dictionary codes (carquet_column_reader_reset_codes)
 *
 * A dictionary-encoded column read as codes into its dictionary, and a column
 * chunk that falls back to PLAIN pages after its dictionary pages (as writers
 * do once a dictionary is full): the read stops at the first PLAIN page with
 * CARQUET_ERROR_INVALID_ENCODING, and carquet_column_reader_materialize() reads
 * its values from there. carquet's writer falls back for whole chunks only, so
 * the mixed chunk is spliced from two files' chunks: a dictionary-encoded one,
 * then a PLAIN one's data pages, under the first's footer, grown to both.
 */

#include <carquet/carquet.h>
#include "test_helpers.h"
#include "core/arena.h"
#include "core/buffer.h"
#include "thrift/parquet_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DICT_ROWS 3000
#define PLAIN_ROWS 2000
#define DICT_VALUES 7

static int64_t dict_value(int64_t i) { return 100 + i % DICT_VALUES; }
static int64_t plain_value(int64_t i) { return 1000000 + i; }

/* one INT64 REQUIRED column v, uncompressed, without statistics: rows of
 * value(i), dictionary-encoded or PLAIN */
static int write_file(const char* path, int64_t rows, int64_t (*value)(int64_t), bool dict) {
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_schema_t* schema = carquet_schema_create(&err);
    if (!schema || carquet_schema_add_column(schema, "v", CARQUET_PHYSICAL_INT64, NULL,
                                             CARQUET_REPETITION_REQUIRED, 0, 0) != CARQUET_OK) {
        return 1;
    }
    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);
    opts.compression = CARQUET_COMPRESSION_UNCOMPRESSED;
    opts.write_statistics = false;
    carquet_writer_t* w = carquet_writer_create(path, schema, &opts, &err);
    if (!w) {
        carquet_schema_free(schema);
        return 1;
    }
    int64_t* vs = malloc(sizeof(int64_t) * (size_t)rows);
    for (int64_t i = 0; i < rows; i++) vs[i] = value(i);
    int rc = (dict && carquet_writer_set_column_encoding(w, 0, CARQUET_ENCODING_RLE_DICTIONARY) != CARQUET_OK) ||
             carquet_writer_write_batch(w, 0, vs, rows, NULL, NULL) != CARQUET_OK;
    rc |= carquet_writer_close(w) != CARQUET_OK;
    free(vs);
    carquet_schema_free(schema);
    return rc;
}

static uint8_t* read_file(const char* path, size_t* size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* b = malloc((size_t)n);
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) {
        free(b);
        b = NULL;
    }
    fclose(f);
    *size = (size_t)n;
    return b;
}

/* a file's footer parsed into arena */
static int parse_footer(const uint8_t* file, size_t size, carquet_arena_t* arena,
                        parquet_file_metadata_t* md) {
    uint32_t len = (uint32_t)file[size - 8] | ((uint32_t)file[size - 7] << 8) |
                   ((uint32_t)file[size - 6] << 16) | ((uint32_t)file[size - 5] << 24);
    carquet_error_t err = CARQUET_ERROR_INIT;
    return parquet_parse_file_metadata(file + size - 8 - len, len, arena, md, &err) != CARQUET_OK;
}

/* the mixed file: the dictionary file's chunk, then the PLAIN file's data
 * pages, under the dictionary file's footer counting both */
static int splice(const char* dict_path, const char* plain_path, const char* out_path) {
    size_t ds = 0, ps = 0;
    uint8_t* d = read_file(dict_path, &ds);
    uint8_t* p = read_file(plain_path, &ps);
    carquet_arena_t arena;
    parquet_file_metadata_t dm, pm;
    int rc = 1;
    if (!d || !p || carquet_arena_init(&arena) != CARQUET_OK) goto out;
    if (parse_footer(d, ds, &arena, &dm) || parse_footer(p, ps, &arena, &pm)) goto out_arena;
    {
        parquet_column_metadata_t* dc = &dm.row_groups[0].columns[0].metadata;
        const parquet_column_metadata_t* pc = &pm.row_groups[0].columns[0].metadata;
        if (!dc->has_dictionary_page_offset || pc->has_dictionary_page_offset) goto out_arena;
        size_t dict_end = (size_t)(dc->dictionary_page_offset + dc->total_compressed_size);
        size_t plain_begin = (size_t)pc->data_page_offset;
        size_t plain_bytes = (size_t)pc->total_compressed_size;

        dc->num_values += pc->num_values;
        dc->total_compressed_size += pc->total_compressed_size;
        dc->total_uncompressed_size += pc->total_uncompressed_size;
        dc->num_encoding_stats = 0;
        dm.row_groups[0].num_rows += pm.row_groups[0].num_rows;
        dm.row_groups[0].total_byte_size += pm.row_groups[0].total_byte_size;
        if (dm.row_groups[0].has_total_compressed_size) {
            dm.row_groups[0].total_compressed_size += pc->total_compressed_size;
        }
        dm.num_rows += pm.num_rows;

        carquet_buffer_t footer;
        carquet_buffer_init(&footer);
        carquet_error_t err = CARQUET_ERROR_INIT;
        FILE* f = fopen(out_path, "wb");
        if (f && parquet_write_file_metadata(&dm, &footer, &err) == CARQUET_OK) {
            size_t n = carquet_buffer_size(&footer);
            uint8_t len[4] = {(uint8_t)n, (uint8_t)(n >> 8), (uint8_t)(n >> 16), (uint8_t)(n >> 24)};
            rc = fwrite(d, 1, dict_end, f) != dict_end ||
                 fwrite(p + plain_begin, 1, plain_bytes, f) != plain_bytes ||
                 fwrite(carquet_buffer_data(&footer), 1, n, f) != n ||
                 fwrite(len, 1, 4, f) != 4 || fwrite("PAR1", 1, 4, f) != 4;
        }
        if (f && fclose(f) != 0) rc = 1;
        carquet_buffer_destroy(&footer);
    }
out_arena:
    carquet_arena_destroy(&arena);
out:
    free(d);
    free(p);
    return rc;
}

int main(void) {
    carquet_init();
    char dict_path[512], plain_path[512], mixed_path[512];
    carquet_test_temp_path(dict_path, sizeof(dict_path), "test_dictionary_codes_dict");
    carquet_test_temp_path(plain_path, sizeof(plain_path), "test_dictionary_codes_plain");
    carquet_test_temp_path(mixed_path, sizeof(mixed_path), "test_dictionary_codes_mixed");
    if (write_file(dict_path, DICT_ROWS, dict_value, true) ||
        write_file(plain_path, PLAIN_ROWS, plain_value, false) ||
        splice(dict_path, plain_path, mixed_path))
        TEST_FAIL("dictionary_codes", "cannot write the files");

    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_reader_t* r = carquet_reader_open(mixed_path, NULL, &err);
    if (!r) TEST_FAIL("dictionary_codes", err.message);
    static int64_t values[DICT_ROWS + PLAIN_ROWS];
    static uint32_t codes[DICT_ROWS + PLAIN_ROWS];

    /* values, as ever: the spliced chunk reads whole */
    carquet_column_reader_t* c = carquet_reader_get_column(r, 0, 0, &err);
    if (!c || carquet_column_read_batch(c, values, DICT_ROWS + PLAIN_ROWS, NULL, NULL) != DICT_ROWS + PLAIN_ROWS)
        TEST_FAIL("dictionary_codes", "values of the mixed chunk");
    for (int64_t i = 0; i < DICT_ROWS + PLAIN_ROWS; i++) {
        if (values[i] != (i < DICT_ROWS ? dict_value(i) : plain_value(i - DICT_ROWS)))
            TEST_FAIL("dictionary_codes", "a value of the mixed chunk");
    }

    /* codes into the dictionary; the dictionary's values */
    bool coded = false;
    if (carquet_column_reader_reset_codes(c, 0, &coded, &err) != CARQUET_OK || !coded)
        TEST_FAIL("dictionary_codes", "reset_codes");
    int32_t count = -1;
    if (carquet_column_dictionary(c, &count) != NULL || count != 0)
        TEST_FAIL("dictionary_codes", "a dictionary before the first read");
    if (carquet_column_read_batch_ex(c, codes, 10, NULL, NULL, &err) != 10)
        TEST_FAIL("dictionary_codes", "the first codes");
    const int64_t* dict = carquet_column_dictionary(c, &count);
    if (!dict || count != DICT_VALUES)
        TEST_FAIL("dictionary_codes", "the dictionary");
    /* in the middle of a page: refused, nothing changed */
    carquet_error_t e = CARQUET_ERROR_INIT;
    if (carquet_column_reader_materialize(c, &e) != CARQUET_ERROR_INVALID_STATE)
        TEST_FAIL("dictionary_codes", "materialize in the middle of a page");

    /* on to the PLAIN page: a partial read of the codes before it, stopped there */
    int64_t got = 10;
    for (;;) {
        carquet_error_t re = CARQUET_ERROR_INIT;
        int64_t n = carquet_column_read_batch_ex(c, codes + got, 700, NULL, NULL, &re);
        if (carquet_error_is_set(&re)) {
            if (re.code != CARQUET_ERROR_INVALID_ENCODING)
                TEST_FAIL("dictionary_codes", re.message);
            got += n > 0 ? n : 0;
            break;
        }
        if (n <= 0) TEST_FAIL("dictionary_codes", "the codes ended before the PLAIN page");
        got += n;
    }
    if (got != DICT_ROWS) TEST_FAIL("dictionary_codes", "not stopped at the PLAIN page");
    for (int64_t i = 0; i < DICT_ROWS; i++) {
        if (codes[i] >= (uint32_t)count || dict[codes[i]] != dict_value(i))
            TEST_FAIL("dictionary_codes", "a code");
    }

    /* the rest as values, from the PLAIN page */
    if (carquet_column_reader_materialize(c, &err) != CARQUET_OK)
        TEST_FAIL("dictionary_codes", "materialize at the PLAIN page");
    if (carquet_column_read_batch_ex(c, values, PLAIN_ROWS, NULL, NULL, &err) != PLAIN_ROWS ||
        carquet_column_has_next(c))
        TEST_FAIL("dictionary_codes", "the PLAIN page's values");
    for (int64_t i = 0; i < PLAIN_ROWS; i++) {
        if (values[i] != plain_value(i))
            TEST_FAIL("dictionary_codes", "a value of the PLAIN page");
    }

    /* a PLAIN chunk: not coded, its values read as ever */
    carquet_reader_t* pr = carquet_reader_open(plain_path, NULL, &err);
    carquet_column_reader_t* pc = pr ? carquet_reader_get_column(pr, 0, 0, &err) : NULL;
    if (!pc || carquet_column_reader_reset_codes(pc, 0, &coded, &err) != CARQUET_OK || coded ||
        carquet_column_read_batch(pc, values, PLAIN_ROWS, NULL, NULL) != PLAIN_ROWS || values[5] != plain_value(5))
        TEST_FAIL("dictionary_codes", "a PLAIN chunk");

    carquet_column_reader_free(pc);
    carquet_reader_close(pr);
    carquet_column_reader_free(c);
    carquet_reader_close(r);
    remove(dict_path);
    remove(plain_path);
    remove(mixed_path);
    TEST_PASS("dictionary_codes");
    printf("\nAll dictionary code tests passed.\n");
    return 0;
}

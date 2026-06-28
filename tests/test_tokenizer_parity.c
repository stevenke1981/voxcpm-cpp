#include "voxcpm.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }

    vcpm_model_params mp = vcpm_default_model_params();
    vcpm_context * ctx = vcpm_load_model(argv[1], &mp);
    assert(ctx != NULL);
    assert(vcpm_model_is_loaded(ctx) && "model must load");

    int32_t ids[16];
    int n = vcpm_tokenize(ctx, "Hello world.", ids, 16);
    assert(n == 3 && "Hello world. should match Python fixture token count");
    assert(ids[0] == 21045);
    assert(ids[1] == 2809);
    assert(ids[2] == 72);

    printf("PASS: tokenizer parity for Hello world. -> [%d, %d, %d]\n",
           ids[0], ids[1], ids[2]);

    const char chinese[] =
        "\xE4\xBD\xA0" "\xE5\xA5\xBD" "\xEF\xBC\x8C"
        "\xE8\xBF\x99" "\xE6\x98\xAF" "\xE6\xB5\x8B"
        "\xE8\xAF\x95" "\xE3\x80\x82";
    const int32_t chinese_expected[] = {
        59320, 59496, 59495, 65, 59431, 59390, 59972, 59865, 66
    };
    n = vcpm_tokenize(ctx, chinese, ids, 16);
    assert(n == 9 && "multi-character Chinese BPE tokens must be expanded");
    for (int i = 0; i < n; ++i) {
        assert(ids[i] == chinese_expected[i]);
    }
    printf("PASS: tokenizer parity for single-character Chinese expansion\n");

    vcpm_free(ctx);
    return 0;
}

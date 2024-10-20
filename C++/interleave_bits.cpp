#include <immintrin.h>
#include <stdio.h>

__m128i interleave_bits(__m128i a, __m128i b, __m128i c, __m128i d) {
	// Shuffle masks for various bit interleaving stages
	const __m128i mask1 = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
	const __m128i mask2 = _mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
	const __m128i mask3 = _mm_set_epi8(15, 7, 14, 6, 13, 5, 12, 4, 11, 3, 10, 2, 9, 1, 8, 0);

	// Interleave the lower 16 bits of each integer
	__m128i ab = _mm_shuffle_epi8(a, mask1);
	__m128i cd = _mm_shuffle_epi8(c, mask1);
	__m128i ab_lo = _mm_and_si128(ab, _mm_set1_epi16(0x00FF));
	__m128i cd_lo = _mm_and_si128(cd, _mm_set1_epi16(0x00FF));
	__m128i abcd_lo = _mm_or_si128(ab_lo, _mm_slli_epi16(cd_lo, 1));

	// Interleave the upper 16 bits of each integer
	__m128i ab_hi = _mm_srli_epi16(ab, 8);
	__m128i cd_hi = _mm_srli_epi16(cd, 8);
	__m128i abcd_hi = _mm_or_si128(ab_hi, _mm_slli_epi16(cd_hi, 1));

	// Combine the lower and upper 16-bit results
	__m128i result = _mm_or_si128(abcd_lo, _mm_slli_epi32(abcd_hi, 2));

	// Continue interleaving
	result = _mm_shuffle_epi8(result, mask2);
	result = _mm_shuffle_epi8(result, mask3);

	return result;
}

int main() {
	unsigned int a = 0x12345678;
	unsigned int b = 0x9ABCDEF0;
	unsigned int c = 0x0FEDCBA9;
	unsigned int d = 0x87654321;

	__m128i input = _mm_set_epi32(a, b, c, d);
	__m128i result = interleave_bits(input, input, input, input);

	unsigned int output[4];
	_mm_storeu_si128((__m128i*)output, result);

	printf("Interleaved result: 0x%08X%08X%08X%08X\n", output[3], output[2], output[1], output[0]);

	return 0;
}
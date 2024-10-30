#include <iostream>
#include <vector>
#include <bitset>

using namespace std;

// Function to read bits from a byte buffer
unsigned int readBits(const vector<uint8_t>& buffer, int& bitPos, int numBits) {
    unsigned int value = 0;
    for (int i = 0; i < numBits; ++i) {
        value <<= 1;
        value |= (buffer[bitPos / 8] >> (7 - (bitPos % 8))) & 1;
        bitPos++;
    }
    return value;
}

// Function to read an unsigned exponential-Golomb-coded integer
unsigned int readUE(const vector<uint8_t>& buffer, int& bitPos) {
    int leadingZeroBits = 0;
    while ((readBits(buffer, bitPos, 1) == 0) && (bitPos < buffer.size() * 8)) {
        leadingZeroBits++;
    }
    return (1 << leadingZeroBits) - 1 + readBits(buffer, bitPos, leadingZeroBits);
}

// Function to parse SPS NAL unit
void parseSPS(const vector<uint8_t>& nalUnit) {
    int bitPos = 8; // Skip NAL unit header

    // Parse SPS data
    int sps_video_parameter_set_id = readBits(nalUnit, bitPos, 4);
    int sps_max_sub_layers_minus1 = readBits(nalUnit, bitPos, 3);
    int sps_temporal_id_nesting_flag = readBits(nalUnit, bitPos, 1);


    cout << "SPS:" << endl;
    cout << "  sps_video_parameter_set_id: " << sps_video_parameter_set_id << endl;
    cout << "  sps_max_sub_layers_minus1: " << sps_max_sub_layers_minus1 << endl;
    cout << "  sps_temporal_id_nesting_flag: " << sps_temporal_id_nesting_flag << endl;
}

// Function to parse PPS NAL unit
void parsePPS(const vector<uint8_t>& nalUnit) {
    int bitPos = 8; // Skip NAL unit header

    // Parse PPS data
    int pps_pic_parameter_set_id = readUE(nalUnit, bitPos);
    int pps_sps_video_parameter_set_id = readUE(nalUnit, bitPos);
    int pps_dependent_slice_segments_enabled_flag = readBits(nalUnit, bitPos, 1);


    cout << "PPS:" << endl;
    cout << "  pps_pic_parameter_set_id: " << pps_pic_parameter_set_id << endl;
    cout << "  pps_sps_video_parameter_set_id: " << pps_sps_video_parameter_set_id << endl;
    cout << "  pps_dependent_slice_segments_enabled_flag: " << pps_dependent_slice_segments_enabled_flag << endl;
}

int main() {
    // Example HEVC NAL units (replace with your actual data)
    vector<uint8_t> spsNalUnit = { 0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00, 0x78, 0xa0, 0x03, 0xc0, 0x80, 0x10, 0x05, 0xba, 0x00, 0x01, 0x00, 0x04, 0x20, 0x00, 0x00, 0x0f, 0x42, 0x01, 0x01 };
    vector<uint8_t> ppsNalUnit = { 0x44, 0x01, 0xc1, 0x73, 0xb8, 0x40 };

    // Parse SPS and PPS
    parseSPS(spsNalUnit);
    parsePPS(ppsNalUnit);

    return 0;
}
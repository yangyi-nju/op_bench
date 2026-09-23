// Original synthetic integration fixture: reduce each row of a dense CPU array.
extern "C" void row_sum(const double* input, int rows, int cols, double* output) {
    for (int row = 0; row < rows; ++row) {
        double total = 0.0;
        for (int col = 0; col < cols; ++col) {
            total += input[col];
        }
        output[row] = total;
    }
}

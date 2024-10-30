__global__ void hadamard_kernel(cuComplex *qstates, int num_qubits) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_qubits) {
    cuComplex q0 = qstates[idx * 2];
    cuComplex q1 = qstates[idx * 2 + 1];
    qstates[idx * 2] = (q0 + q1) / sqrt(2.0f);
    qstates[idx * 2 + 1] = (q0 - q1) / sqrt(2.0f);
  }
}
__global__ void apply_rotation_gate(cuComplex *state_vector, float angle, int qubit_index, int num_qubits) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_qubits) return;

  // Apply rotation gate on target qubit based on index
  if ((idx & (1 << qubit_index)) != 0) {
    int other_idx = idx ^ (1 << qubit_index);
    cuComplex amp1 = state_vector[idx];
    cuComplex amp2 = state_vector[other_idx];
    state_vector[idx] = cuCmulf(amp1, make_cuComplex(cosf(angle), 0.0f)) + cuCmulf(amp2, make_cuComplex(-sinf(angle), 0.0f));
    state_vector[other_idx] = cuCmulf(amp1, make_cuComplex(sinf(angle), 0.0f)) + cuCmulf(amp2, make_cuComplex(cosf(angle), 0.0f));
  }
}


float ndc_depth_to_z(float z, float proj22, float proj32) {
  return -proj32 / (z + proj22);
}

float z_to_ndc_depth(float z, float proj22, float proj32) {
  return -proj22 - proj32 / z;
}

#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(location = 0) in vec4 inPosition;
layout(location = 1) in flat vec4 inNdcRadius;

layout(set = 0, binding = 0) uniform LightBuffer {
  mat4 viewMatrix;
  mat4 viewMatrixInverse;
  mat4 projectionMatrix;
  mat4 projectionMatrixInverse;
  int width;
  int height;
} lightBuffer;

layout(set = 1, binding = 0) uniform ObjectBuffer {
  mat4 modelMatrix;
  mat4 prevModelMatrix;
  uvec4 segmentation;
  float transparency;
  int shadeFlat;
} objectBuffer;

void main() {
  if (gl_PointCoord.s * gl_PointCoord.s + gl_PointCoord.t * gl_PointCoord.t > 1) {
    discard;
  }
  // vec2 centerNdc = inNdcRadius.xy;
  // vec2 res = vec2(lightBuffer.width, lightBuffer.height) * RESOLUTION_SCALE;
  // vec2 pixelNdc = gl_FragCoord.xy / res * 2.0 - 1.0;
  // vec2 offsetNdc = pixelNdc - centerNdc;
  // vec2 offset = offsetNdc * (-inPosition.z) / vec2(lightBuffer.projectionMatrix[0][0], lightBuffer.projectionMatrix[1][1]);
  // float radius = inNdcRadius.w;
  // offset /= radius;
  // if (offset.x * offset.x + offset.y * offset.y > 1) {
  //   discard;
  // }
}

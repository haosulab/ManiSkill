#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

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

layout(location = 0) in vec3 position;
layout(location = 1) in float scale;
layout(location = 2) in vec4 color;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out flat vec4 outNdcRadius;

void main() {
  mat4 modelView = lightBuffer.viewMatrix * objectBuffer.modelMatrix;
  outPosition = modelView * vec4(position, 1);

  float radius = scale;

  gl_PointSize = lightBuffer.projectionMatrix[0][0] * lightBuffer.width * radius;

  gl_Position = lightBuffer.projectionMatrix * outPosition;
  outNdcRadius = vec4(gl_Position.xyz / gl_Position.w, radius);
}

#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(set = 0, binding = 0) uniform CameraBuffer {
  mat4 viewMatrix;
  mat4 projectionMatrix;
  mat4 viewMatrixInverse;
  mat4 projectionMatrixInverse;
  mat4 prevViewMatrix;
  mat4 prevViewMatrixInverse;
  float width;
  float height;
} cameraBuffer;

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
layout(location = 1) out vec4 outPrevPosition;
layout(location = 2) out vec4 outColor;

void main() {
  mat4 modelView = cameraBuffer.viewMatrix * objectBuffer.modelMatrix;
  outPosition = modelView * vec4(position, 1);
  outPrevPosition = cameraBuffer.prevViewMatrix * objectBuffer.prevModelMatrix * vec4(position, 1);
  gl_Position = cameraBuffer.projectionMatrix * outPosition;
  outColor = color;
}

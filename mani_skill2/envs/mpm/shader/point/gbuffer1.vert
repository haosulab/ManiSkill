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
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec3 tangent;
layout(location = 4) in vec3 bitangent;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec2 outUV;
layout(location = 2) out flat uvec4 outSegmentation;
layout(location = 3) out vec3 objectCoord;
layout(location = 4) out mat3 outTbn;

void main() {
  outSegmentation = objectBuffer.segmentation;
  mat4 modelView = cameraBuffer.viewMatrix * objectBuffer.modelMatrix;
  mat3 normalMatrix = mat3(transpose(inverse(modelView)));
  objectCoord = position;
  outPosition = modelView * vec4(position, 1);
  outUV = uv;
  gl_Position = cameraBuffer.projectionMatrix * outPosition;
  vec3 outTangent = normalize(normalMatrix * tangent);
  vec3 outBitangent = normalize(normalMatrix * bitangent);
  vec3 outNormal = normalize(normalMatrix * normal);
  outTbn = mat3(outTangent, outBitangent, outNormal);
}

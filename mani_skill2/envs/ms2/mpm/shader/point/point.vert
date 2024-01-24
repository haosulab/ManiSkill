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

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outCenterRadius;

void main() {
  mat4 modelView = cameraBuffer.viewMatrix * objectBuffer.modelMatrix;
  vec4 outPosition = modelView * vec4(position, 1);
  float radius = scale;

  vec3 v0 = outPosition.xyz + vec3(-radius, -radius, -radius);
  vec3 v1 = outPosition.xyz + vec3(-radius, -radius, radius);
  vec3 v2 = outPosition.xyz + vec3(-radius, radius, -radius);
  vec3 v3 = outPosition.xyz + vec3(-radius, radius, radius);
  vec3 v4 = outPosition.xyz + vec3(radius, -radius, -radius);
  vec3 v5 = outPosition.xyz + vec3(radius, -radius, radius);
  vec3 v6 = outPosition.xyz + vec3(radius, radius, -radius);
  vec3 v7 = outPosition.xyz + vec3(radius, radius, radius);

  vec4 p0 = cameraBuffer.projectionMatrix * vec4(v0, 1.0);
  vec4 p1 = cameraBuffer.projectionMatrix * vec4(v1, 1.0);
  vec4 p2 = cameraBuffer.projectionMatrix * vec4(v2, 1.0);
  vec4 p3 = cameraBuffer.projectionMatrix * vec4(v3, 1.0);
  vec4 p4 = cameraBuffer.projectionMatrix * vec4(v4, 1.0);
  vec4 p5 = cameraBuffer.projectionMatrix * vec4(v5, 1.0);
  vec4 p6 = cameraBuffer.projectionMatrix * vec4(v6, 1.0);
  vec4 p7 = cameraBuffer.projectionMatrix * vec4(v7, 1.0);

  vec2 s0 = p0.xy / p0.w;
  vec2 s1 = p1.xy / p1.w;
  vec2 s2 = p2.xy / p2.w;
  vec2 s3 = p3.xy / p3.w;
  vec2 s4 = p4.xy / p4.w;
  vec2 s5 = p5.xy / p5.w;
  vec2 s6 = p6.xy / p6.w;
  vec2 s7 = p7.xy / p7.w;

  vec2 lower = min(min(min(s0, s1), min(s2, s3)), min(min(s4, s5), min(s6, s7)));
  vec2 upper = max(max(max(s0, s1), max(s2, s3)), max(max(s4, s5), max(s6, s7)));
  vec2 extent_ndc = upper - lower;
  vec2 center_ndc = (lower + upper) / 2.0;

  vec2 extent = extent_ndc * vec2(cameraBuffer.width, cameraBuffer.height) / 2.0;
  float size = max(extent.x, extent.y);

  gl_PointSize = size;
  gl_Position = vec4(center_ndc, 1.0, 1.0);
  outCenterRadius = vec4(outPosition.xyz, radius);

  outColor = color;
}

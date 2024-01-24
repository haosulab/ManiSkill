#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (constant_id = 0) const int NUM_DIRECTIONAL_LIGHTS = 3;
layout (constant_id = 1) const int NUM_POINT_LIGHTS = 10;
layout (constant_id = 2) const int NUM_SPOT_LIGHTS = 10;
layout (constant_id = 3) const float RESOLUTION_SCALE = 1.0;

layout(location = 0) in vec4 inColor;
layout(location = 1) in flat vec4 inCenterRadius;

layout(location = 0) out vec4 outPointColor;
layout(location = 1) out vec4 outPointDepthLinear;

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

void main() {
  vec3 center = inCenterRadius.xyz;
  float radius = inCenterRadius.w;

  vec2 res = vec2(cameraBuffer.width, cameraBuffer.height) * RESOLUTION_SCALE;
  vec2 pixelNdc = gl_FragCoord.xy / res * 2.0 - 1.0;

  vec4 dir4 = cameraBuffer.projectionMatrixInverse * vec4(pixelNdc, 1.0, 1.0);
  vec3 dir = normalize(dir4.xyz / dir4.w);

  float a = dot(dir, center);
  float N = a * a - (dot(center, center) - radius * radius);

  if (N < 0) {
    discard;
  }

  float d = dot(dir, center) - sqrt(N);

  if (d < 0) {
    discard;
  }

  vec4 csPosition = vec4(dir * d, 1.0);
  vec4 ndc = cameraBuffer.projectionMatrix * csPosition;
  ndc /= ndc.w;
  gl_FragDepth = ndc.z;

  outPointColor = inColor;
  outPointDepthLinear.x = csPosition.z;
}

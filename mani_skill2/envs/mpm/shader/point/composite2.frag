#version 450

layout(set = 0, binding = 0) uniform sampler2D samplerLighting;
layout(set = 0, binding = 1) uniform sampler2D samplerLighting1;
layout(set = 0, binding = 2) uniform sampler2D samplerAlbedo2;
layout(set = 0, binding = 3) uniform sampler2D samplerGbufferDepth;
layout(set = 0, binding = 4) uniform sampler2D samplerGbuffer1Depth;
layout(set = 0, binding = 5) uniform sampler2D samplerGbuffer2Depth;
layout(set = 0, binding = 6) uniform usampler2D samplerSegmentation0;
layout(set = 0, binding = 7) uniform usampler2D samplerSegmentation1;
layout(set = 0, binding = 8) uniform sampler2D samplerLineDepth;
layout(set = 0, binding = 9) uniform sampler2D samplerLine;
layout(set = 0, binding = 10) uniform sampler2D samplerSmoothedDepthLinear;
layout(set = 0, binding = 11) uniform sampler2D samplerPoint;

layout(set = 0, binding = 12) uniform sampler2D samplerPosition0;
layout(set = 0, binding = 13) uniform sampler2D samplerPosition1;


layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outDepthLinear;
layout(location = 2) out uvec4 outSegmentation;
layout(location = 3) out vec4 outSegmentationView0;
layout(location = 4) out vec4 outSegmentationView1;
layout(location = 5) out vec4 outPosition;


layout(set = 1, binding = 0) uniform CameraBuffer {
  mat4 viewMatrix;
  mat4 projectionMatrix;
  mat4 viewMatrixInverse;
  mat4 projectionMatrixInverse;
  mat4 prevViewMatrix;
  mat4 prevViewMatrixInverse;
  float width;
  float height;
} cameraBuffer;

vec4 colors[60] = {
  vec4(0.8,  0.4,  0.4 , 1 ),
  vec4(0.8,  0.41, 0.24, 1 ),
  vec4(0.8,  0.75, 0.32, 1 ),
  vec4(0.6,  0.8,  0.4 , 1 ),
  vec4(0.35, 0.8,  0.24, 1 ),
  vec4(0.32, 0.8,  0.51, 1 ),
  vec4(0.4,  0.8,  0.8 , 1 ),
  vec4(0.24, 0.63, 0.8 , 1 ),
  vec4(0.32, 0.37, 0.8 , 1 ),
  vec4(0.6,  0.4,  0.8 , 1 ),
  vec4(0.69, 0.24, 0.8 , 1 ),
  vec4(0.8,  0.32, 0.61, 1 ),
  vec4(0.8,  0.32, 0.32, 1 ),
  vec4(0.8,  0.64, 0.4 , 1 ),
  vec4(0.8,  0.74, 0.24, 1 ),
  vec4(0.56, 0.8,  0.32, 1 ),
  vec4(0.4,  0.8,  0.44, 1 ),
  vec4(0.24, 0.8,  0.46, 1 ),
  vec4(0.32, 0.8,  0.8 , 1 ),
  vec4(0.4,  0.56, 0.8 , 1 ),
  vec4(0.24, 0.3,  0.8 , 1 ),
  vec4(0.56, 0.32, 0.8 , 1 ),
  vec4(0.8,  0.4,  0.76, 1 ),
  vec4(0.8,  0.24, 0.58, 1 ),
  vec4(0.8,  0.24, 0.24, 1 ),
  vec4(0.8,  0.61, 0.32, 1 ),
  vec4(0.72, 0.8,  0.4 , 1 ),
  vec4(0.52, 0.8,  0.24, 1 ),
  vec4(0.32, 0.8,  0.37, 1 ),
  vec4(0.4,  0.8,  0.68, 1 ),
  vec4(0.24, 0.8,  0.8 , 1 ),
  vec4(0.32, 0.51, 0.8 , 1 ),
  vec4(0.48, 0.4,  0.8 , 1 ),
  vec4(0.52, 0.24, 0.8 , 1 ),
  vec4(0.8,  0.32, 0.75, 1 ),
  vec4(0.8,  0.4,  0.52, 1 ),
  vec4(0.8,  0.52, 0.4 , 1 ),
  vec4(0.8,  0.58, 0.24, 1 ),
  vec4(0.7,  0.8,  0.32, 1 ),
  vec4(0.48, 0.8,  0.4 , 1 ),
  vec4(0.24, 0.8,  0.3 , 1 ),
  vec4(0.32, 0.8,  0.66, 1 ),
  vec4(0.4,  0.68, 0.8 , 1 ),
  vec4(0.24, 0.46, 0.8 , 1 ),
  vec4(0.42, 0.32, 0.8 , 1 ),
  vec4(0.72, 0.4,  0.8 , 1 ),
  vec4(0.8,  0.24, 0.74, 1 ),
  vec4(0.8,  0.32, 0.46, 1 ),
  vec4(0.8,  0.46, 0.32, 1 ),
  vec4(0.8,  0.76, 0.4 , 1 ),
  vec4(0.69, 0.8,  0.24, 1 ),
  vec4(0.42, 0.8,  0.32, 1 ),
  vec4(0.4,  0.8,  0.56, 1 ),
  vec4(0.24, 0.8,  0.63, 1 ),
  vec4(0.32, 0.66, 0.8 , 1 ),
  vec4(0.4,  0.44, 0.8 , 1 ),
  vec4(0.35, 0.24, 0.8 , 1 ),
  vec4(0.7,  0.32, 0.8 , 1 ),
  vec4(0.8,  0.4,  0.64, 1 ),
  vec4(0.8,  0.24, 0.41, 1 )
};


void main() {
  float d0 = texture(samplerGbufferDepth, inUV).x;
  float d1 = texture(samplerGbuffer1Depth, inUV).x;
  float d2 = texture(samplerGbuffer2Depth, inUV).x;

  float pointDepth = texture(samplerSmoothedDepthLinear, inUV).x;
  float dp = (cameraBuffer.projectionMatrix[2][2] * pointDepth + cameraBuffer.projectionMatrix[3][2]) / (-pointDepth);

  vec4 pointColor = texture(samplerPoint, inUV);

  vec4 outColor0 = texture(samplerLighting, inUV);
  vec4 outColor1 = texture(samplerLighting1, inUV);
  vec4 outColor2 = texture(samplerAlbedo2, inUV);

  vec4 outPos0 = vec4(texture(samplerPosition0, inUV).xyz, d0);
  vec4 outPos1 = vec4(texture(samplerPosition1, inUV).xyz, d1);

  // depth composite for 0 and 2
  float factor = step(d0, d2);
  float d = min(d0, d2);
  outColor = outColor0 * factor + outColor2 * (1 - factor);
  outPosition = outPos0;

  // depth composite for 02 and p
  if (pointDepth < 0) {
    factor = step(d, dp);
    d = min(d, dp);
    outColor = outColor * factor + pointColor * (1 - factor);

    // convert dp to pointPos
    vec2 res = vec2(cameraBuffer.width, cameraBuffer.height);
    vec2 pixelNdc = gl_FragCoord.xy / res * 2.0 - 1.0;
    vec4 pointPos = cameraBuffer.projectionMatrixInverse * vec4(pixelNdc, dp, 1.0);
    pointPos /= pointPos.w;
    pointPos.w = dp;

    outPosition = outPosition * factor + pointPos * (1 - factor);
  }

  // blend for 02p and 1
  vec3 blend = outColor1.a * outColor1.rgb + (1 - outColor1.a) * outColor.rgb;
  factor = step(d, d1);
  outColor = vec4((1 - factor) * blend + factor * outColor.rgb, 1.f);

  // outPosition = outPos0 * factor + outPos1 * (1 - factor);
  outPosition = outPosition * factor + outPos1 * (1 - factor);

  // TODO: position for points

  // float pointDepth = texture(samplerSmoothedDepthLinear, inUV).x;
  // if (pointDepth < 0 && (pointDepth > outPosition.z || outPosition.z == 0)) {
  //   outColor = vec4(pointColor.xyz, 1);
  // }

  outColor = pow(outColor, vec4(1/2.2, 1/2.2, 1/2.2, 1));

  // vec4 csPosition = cameraBuffer.projectionMatrixInverse * (vec4(inUV * 2 - 1, min(d0, d1), 1));
  outDepthLinear = vec4(vec3(-outPosition.z), 1.);

  uvec4 seg0 = texture(samplerSegmentation0, inUV);
  uvec4 seg1 = texture(samplerSegmentation1, inUV);
  outSegmentation = d0 < d1 ? seg0 : seg1;

  outSegmentationView0 = mix(vec4(0,0,0,1), colors[outSegmentation.x % 60], sign(outSegmentation.x));
  outSegmentationView1 = mix(vec4(0,0,0,1), colors[outSegmentation.y % 60], sign(outSegmentation.y));

  vec4 lineColor = texture(samplerLine, inUV);
  if (texture(samplerLineDepth, inUV).x < 1) {
    outColor = vec4(lineColor.xyz, 1);
  }

  outColor = clamp(outColor, vec4(0), vec4(1));
}

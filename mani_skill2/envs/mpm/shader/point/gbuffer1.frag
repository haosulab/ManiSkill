#version 450

layout (constant_id = 0) const int NUM_DIRECTIONAL_LIGHTS = 3;
layout (constant_id = 1) const int NUM_POINT_LIGHTS = 10;
layout (constant_id = 2) const int NUM_DIRECTIONAL_LIGHT_SHADOWS = 1;
layout (constant_id = 3) const int NUM_POINT_LIGHT_SHADOWS = 3;
layout (constant_id = 4) const int NUM_TEXTURED_LIGHT_SHADOWS = 1;
layout (constant_id = 5) const int NUM_SPOT_LIGHT_SHADOWS = 10;
layout (constant_id = 6) const int NUM_SPOT_LIGHTS = 10;

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

layout(set = 2, binding = 0) uniform MaterialBuffer {
  vec4 emission;
  vec4 baseColor;
  float fresnel;
  float roughness;
  float metallic;
  float transmission;
  float ior;
  float transmissionRoughness;
  int textureMask;
  int padding1;
} materialBuffer;

layout(set = 2, binding = 1) uniform sampler2D colorTexture;
layout(set = 2, binding = 2) uniform sampler2D roughnessTexture;
layout(set = 2, binding = 3) uniform sampler2D normalTexture;
layout(set = 2, binding = 4) uniform sampler2D metallicTexture;
layout(set = 2, binding = 5) uniform sampler2D emissionTexture;

#include "../common/lights.glsl"

layout(set = 3, binding = 0) uniform SceneBuffer {
  vec4 ambientLight;
  DirectionalLight directionalLights[3];
  SpotLight spotLights[10];
  PointLight pointLights[10];
  SpotLight texturedLights[1];
} sceneBuffer;

struct LightBuffer {
  mat4 viewMatrix;
  mat4 viewMatrixInverse;
  mat4 projectionMatrix;
  mat4 projectionMatrixInverse;
  int width;
  int height;
};

layout(set = 3, binding = 1) uniform ShadowBuffer {
  LightBuffer directionalLightBuffers[3];
  LightBuffer spotLightBuffers[10];
  LightBuffer pointLightBuffers[60];
  LightBuffer texturedLightBuffers[1];
} shadowBuffer;

layout(set = 3, binding = 2) uniform samplerCube samplerPointLightDepths[3];
layout(set = 3, binding = 3) uniform sampler2D samplerDirectionalLightDepths[1];
layout(set = 3, binding = 4) uniform sampler2D samplerTexturedLightDepths[1];
layout(set = 3, binding = 5) uniform sampler2D samplerSpotLightDepths[10];
layout(set = 3, binding = 6) uniform sampler2D samplerTexturedLightTextures[1];

layout(set = 3, binding = 7) uniform samplerCube samplerEnvironment;
layout(set = 3, binding = 8) uniform sampler2D samplerBRDFLUT;

vec4 world2camera(vec4 pos) {
  return cameraBuffer.viewMatrix * pos;
}


vec3 getBackgroundColor(vec3 texcoord) {
  texcoord = vec3(-texcoord.y, texcoord.z, -texcoord.x);
  return textureLod(samplerEnvironment, texcoord, 0).rgb;
}

vec3 diffuseIBL(vec3 albedo, vec3 N) {
  N = vec3(-N.y, N.z, -N.x);
  vec3 color = textureLod(samplerEnvironment, N, 5).rgb;
  return color * albedo;
}

vec3 specularIBL(vec3 fresnel, float roughness, vec3 N, vec3 V) {
  float dotNV = max(dot(N, V), 0);
  vec3 R = 2 * dot(N, V) * N - V;
  R = vec3(-R.y, R.z, -R.x);
  vec3 color = textureLod(samplerEnvironment, R, roughness * 5).rgb;
  vec2 envBRDF = texture(samplerBRDFLUT, vec2(roughness, dotNV)).xy;
  return color * (fresnel * envBRDF.x + envBRDF.y);
}





layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec2 inUV;
layout(location = 2) in flat uvec4 inSegmentation;
layout(location = 3) in vec3 objectCoord;
layout(location = 4) in mat3 inTbn;

layout(location = 0) out vec4 outLighting1;
layout(location = 1) out vec4 outNormal1;
layout(location = 2) out uvec4 outSegmentation1;
layout(location = 3) out vec4 outPosition1;

void main() {
  outSegmentation1 = inSegmentation;

  vec4 emission;
  vec4 albedo;
  vec4 frm;

  if ((materialBuffer.textureMask & 16) != 0) {
    emission = texture(emissionTexture, inUV);
  } else {
    emission = materialBuffer.emission;
  }

  if ((materialBuffer.textureMask & 1) != 0) {
    albedo = texture(colorTexture, inUV);
  } else {
    albedo = materialBuffer.baseColor;
  }

  albedo.a *=  (1.f - objectBuffer.transparency);

  if (albedo.a == 0) {
    discard;
  }

  frm.r = materialBuffer.fresnel * 0.08;

  if ((materialBuffer.textureMask & 2) != 0) {
    frm.g = texture(roughnessTexture, inUV).r;
  } else {
    frm.g = materialBuffer.roughness;
  }

  if ((materialBuffer.textureMask & 8) != 0) {
    frm.b = texture(metallicTexture, inUV).r;
  } else {
    frm.b = materialBuffer.metallic;
  }

  if (objectBuffer.shadeFlat == 0) {
    if ((materialBuffer.textureMask & 4) != 0) {
      outNormal1 = vec4(normalize(inTbn * (texture(normalTexture, inUV).xyz * 2 - 1)), 0);
    } else {
      outNormal1 = vec4(normalize(inTbn * vec3(0, 0, 1)), 0);
    }
  } else {
    vec4 fdx = dFdx(inPosition);
    vec4 fdy = dFdy(inPosition);
    vec3 normal = -normalize(cross(fdx.xyz, fdy.xyz));
    outNormal1 = vec4(normal, 0);
  }

  outPosition1 = inPosition;

  float specular = frm.x;
  float roughness = frm.y;
  float metallic = frm.z;

  vec3 normal = outNormal1.xyz;
  vec4 csPosition = inPosition;
  csPosition /= csPosition.w;

  vec3 camDir = -normalize(csPosition.xyz);

  vec3 diffuseAlbedo = albedo.rgb * (1 - metallic);
  vec3 fresnel = specular * (1 - metallic) + albedo.rgb * metallic;

  vec3 color = emission.rgb;

  for (int i = 0; i < NUM_POINT_LIGHTS; i++) {
    vec3 pos = world2camera(vec4(sceneBuffer.pointLights[i].position.xyz, 1.f)).xyz;
    vec3 l = pos - csPosition.xyz;
    color += computePointLight(
        sceneBuffer.pointLights[i].emission.rgb,
        l, normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  for (int i = 0; i < NUM_DIRECTIONAL_LIGHTS; ++i) {
    color += computeDirectionalLight(
        mat3(cameraBuffer.viewMatrix) * sceneBuffer.directionalLights[i].direction.xyz,
        sceneBuffer.directionalLights[i].emission.rgb,
        normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  for (int i = 0; i < NUM_SPOT_LIGHTS; ++i) {
    vec3 pos = world2camera(vec4(sceneBuffer.spotLights[i].position.xyz, 1.f)).xyz;
    vec3 l = pos - csPosition.xyz;
    vec3 centerDir = mat3(cameraBuffer.viewMatrix) * sceneBuffer.spotLights[i].direction.xyz;
    color += computeSpotLight(
        sceneBuffer.spotLights[i].emission.a,
        sceneBuffer.spotLights[i].direction.a,
        centerDir,
        sceneBuffer.spotLights[i].emission.rgb,
        l, normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  for (int i = 0; i < NUM_TEXTURED_LIGHT_SHADOWS; ++i) {
    mat4 shadowView = shadowBuffer.texturedLightBuffers[i].viewMatrix;
    mat4 shadowProj = shadowBuffer.texturedLightBuffers[i].projectionMatrix;

    vec3 pos = world2camera(vec4(sceneBuffer.texturedLights[i].position.xyz, 1.f)).xyz;
    vec3 centerDir = mat3(cameraBuffer.viewMatrix) * sceneBuffer.texturedLights[i].direction.xyz;
    vec3 l = pos - csPosition.xyz;

    float bias = 0;

    vec4 ssPosition = shadowView * cameraBuffer.viewMatrixInverse * vec4((csPosition.xyz), 1);
    ssPosition.z += bias;
    vec4 shadowMapCoord = shadowProj * ssPosition;
    shadowMapCoord /= shadowMapCoord.w;
    shadowMapCoord.xy = shadowMapCoord.xy * 0.5 + 0.5;

    float resolution = textureSize(samplerTexturedLightDepths[i], 0).x;
    float visibility = texture(samplerTexturedLightTextures[i], shadowMapCoord.xy).x;

    color += visibility * computeSpotLight(
        sceneBuffer.texturedLights[i].emission.a,
        sceneBuffer.texturedLights[i].direction.a,
        centerDir,
        sceneBuffer.texturedLights[i].emission.rgb,
        l, normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  vec3 wnormal = mat3(cameraBuffer.viewMatrixInverse) * normal;
  color += diffuseIBL(diffuseAlbedo, wnormal);
  color += specularIBL(fresnel, roughness,
                       wnormal,
                       mat3(cameraBuffer.viewMatrixInverse) * camDir);

  color += sceneBuffer.ambientLight.rgb * diffuseAlbedo;

  outLighting1 = vec4(color, albedo.a);
}

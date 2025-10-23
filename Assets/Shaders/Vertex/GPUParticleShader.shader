// Assets/Shaders/Vertex/GPUParticleShader.shader
Shader "Unlit/GPUParticleShader"
{
    Properties {}
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }
        LOD 100
        Blend One One // Additive blending for a nice glow effect
        ZWrite Off
        Cull Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct Particle
            {
                float4 lifetime;
                float4 velocity;
                float4 position;
                float4 color;
            };

            StructuredBuffer<Particle> _Particles;
            StructuredBuffer<uint> _AliveIndices;

            struct appdata
            {
                float4 vertex : POSITION;
                uint instanceID : SV_InstanceID;
            };

            struct v2f
            {
                float4 color : COLOR;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                uint p_idx = _AliveIndices[v.instanceID];
                Particle p = _Particles[p_idx];

                // Billboard logic
                float3 center = p.position.xyz;
                float3 viewer = normalize(UnityWorldSpaceViewDir(center));
                float3 up = mul((float3x3)unity_WorldToObject, float3(0,1,0));
                float3 right = cross(viewer, up);

                float lifeRatio = saturate(p.lifetime.x / p.lifetime.y);
                // 根据生命周期调整大小，从大变小
                float size = lerp(2.0, 0.5, lifeRatio);
                float4 pos = float4(center + (v.vertex.x * right + v.vertex.y * up) * size, 1.0);

                o.vertex = UnityObjectToClipPos(pos);
                o.color = p.color;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // 添加一个中心亮、边缘暗的效果，让粒子看起来更像点
                float2 screenPos = i.vertex.xy / i.vertex.w;
                float dist = length(screenPos);
                float falloff = 1.0 - saturate(dist * 0.5);
                return i.color * falloff;
            }
            ENDCG
        }
    }
}
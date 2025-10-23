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

                // Billboard logic to make particles face the camera
                float3 center = p.position.xyz;
                float3 viewer = normalize(UnityWorldSpaceViewDir(center));
                float3 up = mul((float3x3)unity_WorldToObject, float3(0,1,0));
                float3 right = cross(viewer, up);

                float lifeRatio = saturate(p.lifetime.x / p.lifetime.y);
                float size = lerp(0.05, 0.01, lifeRatio); // 恢复原始粒子大小
                float4 pos = float4(center + (v.vertex.x * right + v.vertex.y * up) * size, 1.0);

                o.vertex = UnityObjectToClipPos(pos);
                o.color = float4(p.color.rgb, p.color.a * (1 - lifeRatio)); // 使用粒子实际颜色
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                return i.color;
            }
            ENDCG
        }
    }
}
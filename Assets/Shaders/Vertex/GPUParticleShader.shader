// Assets/Shaders/Vertex/GPUParticleShader.shader
Shader "Unlit/GPUParticleShader"
{
    Properties {
        _ColorOverLifeTex ("Color Over Life (1D)", 2D) = "white" {}
        _SizeOverLifeTex  ("Size Over Life (1D)", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }
        LOD 100
        // 改为预乘Alpha的柔和透明混合
        Blend One OneMinusSrcAlpha
        ZWrite Off
        Cull Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"
            sampler2D _ColorOverLifeTex;
            sampler2D _SizeOverLifeTex;

            // 必须与 Compute/C# 的布局完全一致
            struct Particle
            {
                float4 lifetime;   // x: current, y: max
                float4 velocity;   // xyz
                float4 position;   // xyz in world space
                float4 color;      // rgba
                float4 attributes; // x: emitterId
                uint   globalId;
                uint   padding1;
                uint   padding2;
                uint   padding3;
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
                float visible : TEXCOORD0;
                float2 uv : TEXCOORD1;
                float lifeRatio : TEXCOORD2;
            };

            v2f vert (appdata v)
            {
                v2f o;
                uint p_idx = _AliveIndices[v.instanceID];
                Particle p = _Particles[p_idx];

                // 获取粒子的世界空间位置并变换到视空间
                float3 worldPos = p.position.xyz;
                float4 viewPos = mul(UNITY_MATRIX_V, float4(worldPos, 1.0));
                
                // 标记是否在相机前方（Unity 视空间前方为 z < 0）
                o.visible = viewPos.z <= 0.0 ? 1.0 : 0.0;
                
                // 在视空间构建“速度对齐”的拉伸四边形
                float3 velVS = mul((float3x3)UNITY_MATRIX_V, p.velocity.xyz);
                float velLen = length(velVS);
                float3 dirVS = velLen > 1e-5 ? velVS / velLen : float3(0,1,0);
                float3 rightDir = normalize(cross(dirVS, float3(0,0,1))); // 在屏幕平面上取法向
                float3 upDir = dirVS; // 纵向沿速度方向

                // 根据生命周期从 1D 纹理采样大小
                float lifeRatio = saturate(p.lifetime.x / p.lifetime.y);
                float sizeSample = tex2Dlod(_SizeOverLifeTex, float4(lifeRatio, 0.5, 0, 0)).r;
                float width = max(0.001, sizeSample);
                float lengthScale = saturate(velLen * 0.02);
                float height = max(0.001, sizeSample * (0.5 + lengthScale * 2.0));
                
                // 顶点坐标假设 [-0.5,0.5]
                float3 billboardPosVS = viewPos.xyz + (v.vertex.x * rightDir * width + v.vertex.y * upDir * height);
                
                // 视空间 → 裁剪空间
                float4 clipPos = mul(UNITY_MATRIX_P, float4(billboardPosVS, 1.0));
                
                o.vertex = clipPos;
                o.color = p.color;
                // 传递局部uv（固定到 [0,1]）
                o.uv = v.vertex.xy + 0.5;
                o.lifeRatio = lifeRatio;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // 相机后方的粒子直接丢弃
                if (i.visible < 0.5) clip(-1);
                // 基于局部uv的高斯柔边（不受透视拉伸）
                float2 d = i.uv - 0.5;
                float r2 = dot(d, d);
                float alpha = exp(-r2 * 8.0); // 软硬度可调
                // 生命周期淡入淡出
                float lifeFade = saturate(sin(i.lifeRatio * 3.14159265));
                alpha *= lifeFade;
                // 颜色随时间 1D 纹理
                float4 curveCol = tex2Dlod(_ColorOverLifeTex, float4(i.lifeRatio, 0.5, 0, 0));
                alpha *= curveCol.a;
                // 预乘Alpha输出，配合 Blend One OneMinusSrcAlpha
                float3 rgb = i.color.rgb * curveCol.rgb;
                return float4(rgb * alpha, alpha);
            }
            ENDCG
        }
    }
}
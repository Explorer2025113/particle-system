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
        // Premultiplied alpha soft blending
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

            // Must match Compute/C# layout
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

                // World pos -> view space
                float3 worldPos = p.position.xyz;
                float4 viewPos = mul(UNITY_MATRIX_V, float4(worldPos, 1.0));
                
                // Visible only when in front of camera (Unity view z < 0)
                o.visible = viewPos.z <= 0.0 ? 1.0 : 0.0;
                
                // Velocity-aligned stretched quad in view space
                float3 velVS = mul((float3x3)UNITY_MATRIX_V, p.velocity.xyz);
                float velLen = length(velVS);
                float3 dirVS = velLen > 1e-5 ? velVS / velLen : float3(0,1,0);
                float3 rightDir = normalize(cross(dirVS, float3(0,0,1))); // 在屏幕平面上取法向
                float3 upDir = dirVS; // 纵向沿速度方向

                // Size from 1D texture over life
                float lifeRatio = saturate(p.lifetime.x / p.lifetime.y);
                float sizeSample = tex2Dlod(_SizeOverLifeTex, float4(lifeRatio, 0.5, 0, 0)).r;
                float width = max(0.001, sizeSample);
                float lengthScale = saturate(velLen * 0.02);
                float height = max(0.001, sizeSample * (0.5 + lengthScale * 2.0));
                
                // Local quad assumes [-0.5, 0.5]
                float3 billboardPosVS = viewPos.xyz + (v.vertex.x * rightDir * width + v.vertex.y * upDir * height);
                
                // View -> clip space
                float4 clipPos = mul(UNITY_MATRIX_P, float4(billboardPosVS, 1.0));
                
                o.vertex = clipPos;
                o.color = p.color;
                // Pass local uv in [0,1]
                o.uv = v.vertex.xy + 0.5;
                o.lifeRatio = lifeRatio;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // Discard particles behind camera
                if (i.visible < 0.5) clip(-1);
                // Gaussian soft edge in local uv (not perspective-stretched)
                float2 d = i.uv - 0.5;
                float r2 = dot(d, d);
                float alpha = exp(-r2 * 8.0); // softness
                // Fade in/out over life
                float lifeFade = saturate(sin(i.lifeRatio * 3.14159265));
                alpha *= lifeFade;
                // Sample color over life (1D texture)
                float4 curveCol = tex2Dlod(_ColorOverLifeTex, float4(i.lifeRatio, 0.5, 0, 0));
                alpha *= curveCol.a;
                // Premultiplied alpha output (Blend One OneMinusSrcAlpha)
                float3 rgb = i.color.rgb * curveCol.rgb;
                return float4(rgb * alpha, alpha);
            }
            ENDCG
        }
    }
}
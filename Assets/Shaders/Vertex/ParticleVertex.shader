Shader "Custom/ParticleVertex"
{
    Properties
    {
        _ParticleColor ("Particle Color", Color) = (1,1,1,1)
    }
    
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }
        LOD 100
        
        Blend One One  // 加法混合，让粒子发光
        ZWrite Off
        Cull Off
        
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            
            #include "UnityCG.cginc"
            
            struct appdata
            {
                float4 vertex : POSITION;
            };
            
            struct v2f
            {
                float4 vertex : SV_POSITION;
                float4 color : COLOR;
                float2 uv : TEXCOORD0;
            };
            
            float4 _ParticleColor;
            
            v2f vert (appdata v)
            {
                v2f o;
                
                // 直接使用顶点位置
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.color = _ParticleColor;
                o.uv = v.vertex.xy * 0.5 + 0.5; // 将顶点坐标转换为UV
                
                return o;
            }
            
            fixed4 frag (v2f i) : SV_Target
            {
                // 创建圆形粒子效果
                float2 center = float2(0.5, 0.5);
                float dist = distance(i.uv, center);
                float alpha = 1.0 - smoothstep(0.0, 0.5, dist); // 圆形衰减
                
                // 让粒子发光
                fixed4 color = i.color * alpha;
                color.rgb *= 2.0; // 增强亮度
                
                return color;
            }
            ENDCG
        }
    }
}
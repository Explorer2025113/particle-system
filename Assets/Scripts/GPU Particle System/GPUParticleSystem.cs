using UnityEngine;
using UnityEngine.Rendering.Universal;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

public class GPUParticleSystem : MonoBehaviour
{
    [System.Serializable]
    public struct Emitter { public string name; public bool enabled; public int emissionRate; public Vector3 position; public float radius; public Vector3 initialVelocity; public float minInitialSpeed; public float maxInitialSpeed; public Color color; }
    
    // 旧力场结构已移除

    // 移除多涡旋/环带/丝带模式，改为银河盘模式参数

    [StructLayout(LayoutKind.Sequential)]
    private struct EmitterGPU { public Vector4 position; public Vector4 initialVelocity; public Vector4 speedMinMax; public Color color; public Vector4 ratesAndEnabled; }

    //

    // 无需额外结构

    // （调试系统已移除）
    private int _lastVisiblePingPong = 1; // 与渲染一致的可见Alive列表

    [Header("Emitters")]
    public List<Emitter> emitters = new List<Emitter>();
    [Header("Global Forces")]
    public Vector3 globalForce = Vector3.zero;
    
    [Header("Galactic Disk Mode")] 
    public bool galacticDiskEnabled = true;
    public float diskRadius = 150f;            // 盘最大半径
    public float diskHalfThickness = 3f;       // 盘半厚度（y方向）
    public float centerGravity = 20f;          // 中心引力强度（越大越收紧）
    public float planeDamping = 3f;            // 拉回到盘面的强度
    public float flatRotationSpeed = 25f;      // 外盘趋于平坦的切向速度上限
    public float rotationScaleRadius = 50f;    // 切向速度上升尺度 r0
    public int spiralArms = 3;                 // 螺旋臂数量
    public float spiralTightness = 1.2f;       // 对数螺旋紧致度 b
    public float spiralForce = 6f;             // 沿对数螺旋的压缩力强度
    public float tangentialAlign = 0.06f;      // 速度向切向对齐的混合系数
    public float edgeOutflowStrength = 1.2f;   // 外缘离心外飘强度
    public float edgeOutflowExponent = 2.0f;   // 外缘权重指数（越大越只作用在边缘）
    [Header("Galactic Colors")]
    public bool radialColorEnabled = true;
    public Color coreColor = new Color(1.0f,0.95f,0.7f);     // 暖白黄
    public Color midColor = new Color(0.75f,0.7f,0.9f);      // 淡紫灰
    public Color outerColor = new Color(0.45f,0.5f,0.65f);   // 冷蓝灰
    
    [Header("Simulation Settings")]
    public float minLifetime = 5.0f;
    public float maxLifetime = 8.0f;
    public float drag = 0.2f;
    [Header("Color Settings")]
    public bool colorOverLife = false; // 关闭颜色统一化，保留发射器原色
    public bool velocityToColor = true; // 启用速度着色
    public float maxSpeedForColor = 20.0f;
    [Header("General Settings")]
     public bool prewarm = true;   // 启用预热，启动即有粒子
    public float prewarmTime = 10.0f;
    public Vector3 renderBounds = new Vector3(100, 100, 100);
    [Header("References")]
    public ComputeShader computeShader;
    public Material particleMaterial;
    public Mesh particleMesh;
    [Header("Curves (1D Textures)")]
    public Texture2D colorOverLifeTex;
    public Texture2D sizeOverLifeTex;

    [Header("Camera Overrides (Runtime)")]
    public bool forceBlackBackground = true;
    public Color backgroundColor = Color.black;
    public bool disablePostProcessing = true;
    public bool autoFrameCameraOnStart = true;
    public float cameraElevationDeg = 25f;   // 仰角（0=俯视盘面，90=从正上）
    public float cameraYawDeg = 30f;         // 水平旋转
    public float cameraMargin = 1.15f;       // 画面边缘留白

    // 旧 Ring/Flow 参数组已移除

    private int kernelUpdateArgs, kernelEmit, kernelSimulate;
    private GraphicsBuffer particleBuffer, deadPoolBuffer, aliveIndicesBufferA, aliveIndicesBufferB, countersBuffer, indirectArgsBuffer;
    private GraphicsBuffer globalIdCounterBuffer;  // 全局ID计数器缓冲区
    private GraphicsBuffer emittersBuffer;
    private GraphicsBuffer[] aliveIndicesBuffers;
    private int pingPongA = 1, pingPongB = 2;
    private const int THREAD_COUNT_1D = 256;

    void OnEnable() 
    { 
        InitializeBuffers(); 
        InitializeParticles(); 
        SetupCamera(); 
        // 无默认随机配置
        if (prewarm) 
        { 
            float fixedDeltaTime = 1.0f / 30.0f; 
            int prewarmSteps = Mathf.CeilToInt(prewarmTime / fixedDeltaTime); 
            for (int i = 0; i < prewarmSteps; i++) 
            { 
                SetShaderParameters(); 
                RunSimulationStep(fixedDeltaTime); 
            } 
        } 
    }
    void OnDisable() { ReleaseBuffers(); }

void Update() 
{ 
    SetShaderParameters(); 
    
    var didSimulate = RunSimulationStep(Time.deltaTime);
    
    
    Bounds bounds = new Bounds(Vector3.zero, renderBounds); 
    particleMaterial.SetBuffer("_Particles", particleBuffer); 
    // 渲染使用与当前可见列表一致的 alive 索引：
    // 若本帧执行了 Simulate，则可见列表已在 B；否则仍在 A
    var visiblePingPong = didSimulate ? pingPongB : pingPongA;
    _lastVisiblePingPong = visiblePingPong;
    particleMaterial.SetBuffer("_AliveIndices", aliveIndicesBuffers[visiblePingPong - 1]); 
    
    if (colorOverLifeTex) particleMaterial.SetTexture("_ColorOverLifeTex", colorOverLifeTex);
    if (sizeOverLifeTex)  particleMaterial.SetTexture("_SizeOverLifeTex", sizeOverLifeTex);
    Graphics.DrawMeshInstancedIndirect(particleMesh, 0, particleMaterial, bounds, indirectArgsBuffer); 
    }
    
    void SetShaderParameters()
    {
        computeShader.SetFloat("_MinLifetime", minLifetime);
        computeShader.SetFloat("_MaxLifetime", maxLifetime);
        computeShader.SetFloat("_Drag", drag);
        computeShader.SetVector("_GlobalForce", globalForce);
        // —— 删除 Curl/Multi/Ring/Flow ——
        // 已还原：不再下发 OpenGL 对齐的粘性/重力/常量速度参数
        computeShader.SetBool("_ColorOverLife", colorOverLife);
        computeShader.SetBool("_VelocityToColor", velocityToColor);
        computeShader.SetFloat("_MaxSpeedForColor", maxSpeedForColor);

        // Galactic Disk uniforms
        computeShader.SetBool("_GalacticDiskEnabled", galacticDiskEnabled);
        computeShader.SetFloat("_DiskRadius", diskRadius);
        computeShader.SetFloat("_DiskHalfThickness", diskHalfThickness);
        computeShader.SetFloat("_CenterGravity", centerGravity);
        computeShader.SetFloat("_PlaneDamping", planeDamping);
        computeShader.SetFloat("_FlatRotationSpeed", flatRotationSpeed);
        computeShader.SetFloat("_RotationScaleRadius", rotationScaleRadius);
        computeShader.SetInt("_SpiralArms", spiralArms);
        computeShader.SetFloat("_SpiralTightness", spiralTightness);
        computeShader.SetFloat("_SpiralForce", spiralForce);
        computeShader.SetFloat("_TangentialAlign", tangentialAlign);
        computeShader.SetFloat("_EdgeOutflowStrength", edgeOutflowStrength);
        computeShader.SetFloat("_EdgeOutflowExponent", edgeOutflowExponent);
        // Radial color
        computeShader.SetBool("_RadialColorEnabled", radialColorEnabled);
        computeShader.SetVector("_CoreColor", (Vector4)coreColor);
        computeShader.SetVector("_MidColor", (Vector4)midColor);
        computeShader.SetVector("_OuterColor", (Vector4)outerColor);

        if (emitters.Count > 0) { var gpuEmitters = emitters.Select(e => new EmitterGPU { position = new Vector4(e.position.x, e.position.y, e.position.z, e.radius), initialVelocity = e.initialVelocity, speedMinMax = new Vector4(e.minInitialSpeed, e.maxInitialSpeed, 0, 0), color = e.color, ratesAndEnabled = new Vector4(e.emissionRate, e.enabled ? 1.0f : 0.0f, 0, 0) }).ToArray(); emittersBuffer.SetData(gpuEmitters); }
    }
    
    bool RunSimulationStep(float deltaTime) 
    { 
        computeShader.SetBuffer(kernelEmit, "_Emitters", emittersBuffer); 
        // 不再需要力场与涡旋缓冲
        computeShader.SetBuffer(kernelUpdateArgs, "_Counters", countersBuffer); 
        computeShader.SetBuffer(kernelUpdateArgs, "_IndirectArgs", indirectArgsBuffer); 
        computeShader.SetBuffer(kernelEmit, "_Particles", particleBuffer); 
        computeShader.SetBuffer(kernelEmit, "_DeadPool", deadPoolBuffer); 
        computeShader.SetBuffer(kernelEmit, "_Counters", countersBuffer);
        computeShader.SetBuffer(kernelEmit, "_GlobalIdCounter", globalIdCounterBuffer); 
        computeShader.SetBuffer(kernelSimulate, "_Particles", particleBuffer); 
        computeShader.SetBuffer(kernelSimulate, "_DeadPool", deadPoolBuffer); 
        computeShader.SetBuffer(kernelSimulate, "_Counters", countersBuffer); 
        computeShader.SetBuffer(kernelSimulate, "_IndirectArgs", indirectArgsBuffer); 
        computeShader.SetInt("_PingPong_A", pingPongA); 
        computeShader.SetInt("_PingPong_B", pingPongB); 

        computeShader.Dispatch(kernelUpdateArgs, 1, 1, 1); 
        
        // （调试索引重置已移除）

        // =================================================================================
        //     【核心修改】先执行所有Emit，确保初始位置被记录
        // =================================================================================
        for (int i = 0; i < emitters.Count; i++) 
        { 
            if (!emitters[i].enabled) continue; 
            int emissionCount = Mathf.RoundToInt(emitters[i].emissionRate * deltaTime); 
            if (emissionCount > 0) 
            { 
                computeShader.SetInt("_EmitterIndex", i); 
                computeShader.SetInt("_EmissionCount", emissionCount); 
                computeShader.SetVector("_Seeds", new Vector3(Random.value, Random.value, Random.value)); 
                computeShader.SetBuffer(kernelEmit, "_AliveIndices_A", aliveIndicesBuffers[pingPongA - 1]); 
                computeShader.SetInt("_PingPong_A", pingPongA); 
                int emitThreadGroups = Mathf.CeilToInt((float)emissionCount / THREAD_COUNT_1D); 
                computeShader.Dispatch(kernelEmit, emitThreadGroups, 1, 1); 
            } 
        } 
        
        // 读取当前计数
        uint[] currentCounters = new uint[3]; 
        countersBuffer.GetData(currentCounters); 
        uint simulationCount = currentCounters[pingPongA]; 
        bool simulateExecuted = false;
        
        // 有粒子即执行模拟（删除首帧特殊路径）
        if (simulationCount > 0) 
        { 
            computeShader.SetBuffer(kernelSimulate, "_AliveIndices_A", aliveIndicesBuffers[pingPongA - 1]); 
            computeShader.SetBuffer(kernelSimulate, "_AliveIndices_B", aliveIndicesBuffers[pingPongB - 1]); 
            computeShader.SetFloat("_DeltaTime", deltaTime); 
            computeShader.SetInt("_PingPong_A", pingPongA); 
            computeShader.SetInt("_PingPong_B", pingPongB); 
            int simulateThreadGroups = Mathf.CeilToInt((float)simulationCount / THREAD_COUNT_1D); 
            computeShader.Dispatch(kernelSimulate, simulateThreadGroups, 1, 1); 
            simulateExecuted = true;
        } 
        if (simulateExecuted)
        {
            int temp = pingPongA; 
            pingPongA = pingPongB; 
            pingPongB = temp; 
        }

        return simulateExecuted;
    }
    
    void InitializeBuffers() 
    { 
        particleBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ParticleConstants.MAX_PARTICLES, Marshal.SizeOf<Particle>()); 
        deadPoolBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ParticleConstants.MAX_PARTICLES, sizeof(uint)); 
        aliveIndicesBufferA = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ParticleConstants.MAX_PARTICLES, sizeof(uint)); 
        aliveIndicesBufferB = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ParticleConstants.MAX_PARTICLES, sizeof(uint)); 
        aliveIndicesBuffers = new GraphicsBuffer[] { aliveIndicesBufferA, aliveIndicesBufferB }; 
        countersBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 3, sizeof(uint)); 
        indirectArgsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 1, 5 * sizeof(uint)); 
        globalIdCounterBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 2, sizeof(uint));  // 全局ID计数器[0]=粒子ID, [1]=调试索引
        emittersBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Mathf.Max(1, emitters.Count), Marshal.SizeOf<EmitterGPU>()); 
        //

        // （调试缓冲区初始化已移除）
    }

    void InitializeParticles() 
    { 
        kernelUpdateArgs = computeShader.FindKernel("UpdateArgs"); 
        kernelEmit = computeShader.FindKernel("Emit"); 
        kernelSimulate = computeShader.FindKernel("Simulate"); 
        Particle[] initialParticles = new Particle[ParticleConstants.MAX_PARTICLES];
        // 2. 用默认值填充它 (可选，但好习惯)
        for (int i = 0; i < ParticleConstants.MAX_PARTICLES; i++)
        {
            initialParticles[i] = new Particle
            {
                lifetime = Vector4.zero,
                velocity = Vector4.zero,
                position = Vector4.zero,
                color = Vector4.zero,
                attributes = new Vector4(9999, 0, 0, 0) 
            };
        }
        // 3. 将这个干净的数组数据上传到GPU缓冲区
    particleBuffer.SetData(initialParticles);
        uint[] deadIndices = new uint[ParticleConstants.MAX_PARTICLES]; 
        for (uint i = 0; i < ParticleConstants.MAX_PARTICLES; i++) 
        { 
            deadIndices[i] = i; 
        } 
        deadPoolBuffer.SetData(deadIndices); 
        uint[] counters = new uint[] { (uint)ParticleConstants.MAX_PARTICLES, 0, 0 }; 
        countersBuffer.SetData(counters); 
        uint[] globalIdCounter = new uint[] { 0, 0 };  // [0]=粒子ID, [1]=调试索引
        globalIdCounterBuffer.SetData(globalIdCounter);
        uint[] args = new uint[5] { 0, 0, 0, 0, 0 }; 
        if (particleMesh != null) 
            args[0] = particleMesh.GetIndexCount(0); 
        indirectArgsBuffer.SetData(args); 
    }
    void SetupCamera() 
    { 
        var cam = Camera.main; 
        if (cam == null) return; 
        if (forceBlackBackground) 
        { 
            cam.clearFlags = CameraClearFlags.SolidColor; 
            cam.backgroundColor = backgroundColor; 
        } 
        if (disablePostProcessing) 
        { 
            var urpData = cam.GetComponent<UniversalAdditionalCameraData>(); 
            if (urpData != null) urpData.renderPostProcessing = false; 
        }

        if (autoFrameCameraOnStart)
        {
            FrameCameraToDisk(cam);
        }
    }

    [ContextMenu("Frame Camera To Disk")]
    void CM_FrameCamera() { var cam = Camera.main; if (cam!=null) FrameCameraToDisk(cam); }

    void FrameCameraToDisk(Camera cam)
    {
        Vector3 center = Vector3.zero; // 我们的盘心在世界原点
        float R = Mathf.Max(10f, diskRadius);
        float fovRad = cam.fieldOfView * Mathf.Deg2Rad;
        float distance = (R * cameraMargin) / Mathf.Max(0.0001f, Mathf.Tan(fovRad * 0.5f));
        Quaternion rot = Quaternion.Euler(cameraElevationDeg, cameraYawDeg, 0f);
        Vector3 offset = rot * (Vector3.back * distance);
        cam.transform.position = center + offset;
        cam.transform.rotation = Quaternion.LookRotation(center - cam.transform.position, Vector3.up);
        cam.nearClipPlane = 0.1f;
        cam.farClipPlane = Mathf.Max(cam.farClipPlane, R * 6f);
    }

    void ReleaseBuffers() 
    { 
        particleBuffer?.Release(); 
        deadPoolBuffer?.Release(); 
        aliveIndicesBufferA?.Release(); 
        aliveIndicesBufferB?.Release(); 
        countersBuffer?.Release(); 
        indirectArgsBuffer?.Release(); 
        globalIdCounterBuffer?.Release();  // 释放全局ID计数器缓冲区
        emittersBuffer?.Release(); 
        //
        // （调试缓冲区释放已移除）
    }
    
    // （调试方法已移除）
}

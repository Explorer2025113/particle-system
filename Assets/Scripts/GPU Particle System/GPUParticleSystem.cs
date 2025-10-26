using UnityEngine;
using UnityEngine.Rendering.Universal;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

public class GPUParticleSystem : MonoBehaviour
{
    [System.Serializable]
    public struct Emitter { public string name; public bool enabled; public int emissionRate; public Vector3 position; public float radius; public Vector3 initialVelocity; public float minInitialSpeed; public float maxInitialSpeed; public Color color; }
    
    [System.Serializable]
    public struct ForceField { public string name; public bool enabled; public Vector3 position; public float radius; public float strength; }

    [StructLayout(LayoutKind.Sequential)]
    private struct EmitterGPU { public Vector4 position; public Vector4 initialVelocity; public Vector4 speedMinMax; public Color color; public Vector4 ratesAndEnabled; }

    [StructLayout(LayoutKind.Sequential)]
    private struct ForceFieldGPU { public Vector4 positionAndRadius; public Vector4 strengthAndEnabled; }

    // （调试系统已移除）
    private int _lastVisiblePingPong = 1; // 与渲染一致的可见Alive列表

    [Header("Emitters")]
    public List<Emitter> emitters = new List<Emitter>();
    [Header("Global Forces")]
    public List<ForceField> forceFields = new List<ForceField>();
    public Vector3 globalForce = Vector3.zero;
    [Header("Curl Noise (Vortex) Settings")]
    public bool curlNoiseEnabled = true;
    public float curlNoiseScale = 0.05f;
    public float curlNoiseStrength = 3.0f; // 柔和一些
    
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

    [Header("Ring Orbit Mode")]
    public bool ringOrbitEnabled = false;
    public Vector3 ringCenter = Vector3.zero;
    public Vector3 ringAxis = Vector3.up;
    public float ringRadius = 20f;
    public float ringThickness = 4f;
    public float ringGravity = 50f; // 近似万有引力常数（越大越紧密）
    public float ringPlaneDamping = 2.0f; // 拉回到环平面的强度
    public float orbitSpeedScale = 1.0f; // 速度缩放（>1 更快）

    [Header("Flow Ribbon (Nebula) Mode")]
    public bool flowRibbonEnabled = false;
    public Vector3 flowDirection = new Vector3(1, 0, 0);
    public float flowNoiseScale = 0.03f;
    public float flowNoiseStrength = 3.0f;
    public float flowConfinement = 0.0f; // 预留，可用于带状约束
    public float flowDragBoost = 0.0f;   // 预留，可额外增加阻尼

    private int kernelUpdateArgs, kernelEmit, kernelSimulate;
    private GraphicsBuffer particleBuffer, deadPoolBuffer, aliveIndicesBufferA, aliveIndicesBufferB, countersBuffer, indirectArgsBuffer;
    private GraphicsBuffer globalIdCounterBuffer;  // 全局ID计数器缓冲区
    private GraphicsBuffer emittersBuffer, forceFieldsBuffer;
    private GraphicsBuffer[] aliveIndicesBuffers;
    private int pingPongA = 1, pingPongB = 2;
    private const int THREAD_COUNT_1D = 256;

    void OnEnable() 
    { 
        InitializeBuffers(); 
        InitializeParticles(); 
        SetupCamera(); 
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
        computeShader.SetBool("_CurlNoiseEnabled", curlNoiseEnabled);
        computeShader.SetFloat("_CurlNoiseScale", curlNoiseScale);
        computeShader.SetFloat("_CurlNoiseStrength", curlNoiseStrength);
        // 已还原：不再下发 OpenGL 对齐的粘性/重力/常量速度参数
        computeShader.SetBool("_ColorOverLife", colorOverLife);
        computeShader.SetBool("_VelocityToColor", velocityToColor);
        computeShader.SetFloat("_MaxSpeedForColor", maxSpeedForColor);

        // Ring Orbit uniforms
        computeShader.SetBool("_RingOrbitEnabled", ringOrbitEnabled);
        computeShader.SetVector("_RingCenter", ringCenter);
        computeShader.SetVector("_RingAxis", ringAxis);
        computeShader.SetFloat("_RingRadius", ringRadius);
        computeShader.SetFloat("_RingThickness", ringThickness);
        computeShader.SetFloat("_RingGravity", ringGravity);
        computeShader.SetFloat("_RingPlaneDamping", ringPlaneDamping);
        computeShader.SetFloat("_OrbitSpeedScale", orbitSpeedScale);

        // Flow Ribbon uniforms
        computeShader.SetBool("_FlowRibbonEnabled", flowRibbonEnabled);
        computeShader.SetVector("_FlowDirection", flowDirection);
        computeShader.SetFloat("_FlowNoiseScale", flowNoiseScale);
        computeShader.SetFloat("_FlowNoiseStrength", flowNoiseStrength);
        computeShader.SetFloat("_FlowConfinement", flowConfinement);
        computeShader.SetFloat("_FlowDragBoost", flowDragBoost);

        if (emitters.Count > 0) { var gpuEmitters = emitters.Select(e => new EmitterGPU { position = new Vector4(e.position.x, e.position.y, e.position.z, e.radius), initialVelocity = e.initialVelocity, speedMinMax = new Vector4(e.minInitialSpeed, e.maxInitialSpeed, 0, 0), color = e.color, ratesAndEnabled = new Vector4(e.emissionRate, e.enabled ? 1.0f : 0.0f, 0, 0) }).ToArray(); emittersBuffer.SetData(gpuEmitters); }
        if (forceFields.Count > 0) { var gpuForceFields = forceFields.Select(f => new ForceFieldGPU { positionAndRadius = new Vector4(f.position.x, f.position.y, f.position.z, f.radius), strengthAndEnabled = new Vector4(f.strength, f.enabled ? 1.0f : 0.0f, 0, 0) }).ToArray(); forceFieldsBuffer.SetData(gpuForceFields); }
        computeShader.SetInt("_ForceFieldCount", forceFields.Count);
    }
    
    bool RunSimulationStep(float deltaTime) 
    { 
        computeShader.SetBuffer(kernelEmit, "_Emitters", emittersBuffer); 
        computeShader.SetBuffer(kernelSimulate, "_ForceFields", forceFieldsBuffer); 
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
        forceFieldsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Mathf.Max(1, forceFields.Count), Marshal.SizeOf<ForceFieldGPU>()); 

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
        forceFieldsBuffer?.Release(); 
        // （调试缓冲区释放已移除）
    }
    
    // （调试方法已移除）
}

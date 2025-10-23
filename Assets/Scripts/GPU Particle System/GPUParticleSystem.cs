using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class GPUParticleSystem : MonoBehaviour
{
    // Emitter, ForceField, EmitterGPU, ForceFieldGPU 结构体定义 (无变化)
    [System.Serializable]
    public struct Emitter { public string name; public bool enabled; public int emissionRate; public Vector3 position; public float radius; public Vector3 initialVelocity; public float minInitialSpeed; public float maxInitialSpeed; public Color color; }
    [System.Serializable]
    public struct ForceField { public string name; public bool enabled; public Vector3 position; public float radius; public float strength; }
    private struct EmitterGPU { public int enabled; public int emissionRate; public Vector3 position; public float radius; public Vector3 initialVelocity; public float minInitialSpeed; public float maxInitialSpeed; public Color color; }
    private struct ForceFieldGPU { public int enabled; public Vector3 position; public float radius; public float strength; }

    [Header("Emitters")]
    public List<Emitter> emitters = new List<Emitter>();

    [Header("Global Forces")]
    public List<ForceField> forceFields = new List<ForceField>();
    public Vector3 globalForce = Vector3.zero;

    [Header("Curl Noise (Vortex) Settings")]
    public bool curlNoiseEnabled = true;
    public float curlNoiseScale = 1.0f;
    public float curlNoiseStrength = 1.0f;

    [Header("Simulation Settings")]
    public float minLifetime = 10.0f;
    public float maxLifetime = 15.0f;
    public float drag = 0.1f;

    // 【新增】颜色控制
    [Header("Color Settings")]
    public bool colorOverLife = false; // 默认关闭，因为它会导致变黄
    public bool velocityToColor = true; // 默认开启，以获得动态颜色
    public float maxSpeedForColor = 20.0f; // 用于映射速度到颜色的最大速度值

    [Header("General Settings")]
    public bool prewarm = true;
    public float prewarmTime = 10.0f;
    public Vector3 renderBounds = new Vector3(200, 200, 200);

    [Header("References")]
    public ComputeShader computeShader;
    public Material particleMaterial;
    public Mesh particleMesh;

    // 私有变量 (无变化)
    private int kernelUpdateArgs, kernelEmit, kernelSimulate;
    private GraphicsBuffer particleBuffer, deadPoolBuffer, aliveIndicesBufferA, aliveIndicesBufferB, countersBuffer, indirectArgsBuffer;
    private GraphicsBuffer emittersBuffer, forceFieldsBuffer;
    private GraphicsBuffer[] aliveIndicesBuffers;
    private int pingPongA = 1, pingPongB = 2;
    private const int THREAD_COUNT_1D = 256;

    // OnEnable, OnDisable, Update, RunSimulationStep (无变化)
    void OnEnable() { 
        // 如果没有发射器，添加一个默认的
        if (emitters.Count == 0) {
            emitters.Add(new Emitter {
                name = "Default Emitter",
                enabled = true,
                emissionRate = 50, // 进一步降低发射率
                position = Vector3.zero,
                radius = 2.0f, // 增大发射半径
                initialVelocity = Vector3.up * 3.0f + Vector3.right * 1.0f, // 添加一些水平速度
                minInitialSpeed = 2.0f,
                maxInitialSpeed = 5.0f,
                color = Color.cyan // 使用青色作为基础颜色
            });
        }
        
        // 禁用所有力场，避免粒子被吸引到中心
        for (int i = 0; i < forceFields.Count; i++) {
            var ff = forceFields[i];
            ff.enabled = false;
            forceFields[i] = ff;
            Debug.Log($"🔧 禁用力场: {ff.name}");
        }
        
        // 如果没有力场，添加一个禁用的默认力场
        if (forceFields.Count == 0) {
            forceFields.Add(new ForceField {
                name = "Disabled Force Field",
                enabled = false,
                position = Vector3.zero,
                radius = 10.0f,
                strength = 0.0f
            });
            Debug.Log("🔧 添加了禁用的默认力场");
        }
        
        InitializeBuffers(); 
        InitializeParticles(); 
        SetupCamera(); 
        if (prewarm) { 
            Debug.Log("🔥 Pre-warming particle system..."); 
            float fixedDeltaTime = 1.0f / 30.0f; 
            int prewarmSteps = Mathf.CeilToInt(prewarmTime / fixedDeltaTime); 
            for (int i = 0; i < prewarmSteps; i++) { 
                SetShaderParameters(); 
                RunSimulationStep(fixedDeltaTime); 
            } 
            Debug.Log("🔥 Pre-warm complete."); 
        } 
    }
    void OnDisable() { ReleaseBuffers(); }
    void Update() { SetShaderParameters(); RunSimulationStep(Time.deltaTime); Bounds bounds = new Bounds(Vector3.zero, renderBounds); particleMaterial.SetBuffer("_Particles", particleBuffer); particleMaterial.SetBuffer("_AliveIndices", aliveIndicesBuffers[pingPongA - 1]); Graphics.DrawMeshInstancedIndirect(particleMesh, 0, particleMaterial, bounds, indirectArgsBuffer); if (Time.frameCount % 60 == 0) { uint[] finalCounters = new uint[3]; countersBuffer.GetData(finalCounters); Debug.Log($"🔥 GPU粒子系统运行中 - 活粒子: {finalCounters[pingPongA]}"); } }
    private void RunSimulationStep(float deltaTime) { computeShader.SetBuffer(kernelEmit, "_Emitters", emittersBuffer); computeShader.SetBuffer(kernelSimulate, "_ForceFields", forceFieldsBuffer); computeShader.SetBuffer(kernelUpdateArgs, "_Counters", countersBuffer); computeShader.SetBuffer(kernelUpdateArgs, "_IndirectArgs", indirectArgsBuffer); computeShader.SetBuffer(kernelEmit, "_Particles", particleBuffer); computeShader.SetBuffer(kernelEmit, "_DeadPool", deadPoolBuffer); computeShader.SetBuffer(kernelEmit, "_Counters", countersBuffer); computeShader.SetBuffer(kernelSimulate, "_Particles", particleBuffer); computeShader.SetBuffer(kernelSimulate, "_DeadPool", deadPoolBuffer); computeShader.SetBuffer(kernelSimulate, "_Counters", countersBuffer); computeShader.SetBuffer(kernelSimulate, "_IndirectArgs", indirectArgsBuffer); computeShader.SetInt("_PingPong_A", pingPongA); computeShader.SetInt("_PingPong_B", pingPongB); computeShader.Dispatch(kernelUpdateArgs, 1, 1, 1); for (int i = 0; i < emitters.Count; i++) { if (!emitters[i].enabled) continue; int emissionCount = Mathf.RoundToInt(emitters[i].emissionRate * deltaTime); if (emissionCount > 0) { computeShader.SetInt("_EmitterIndex", i); computeShader.SetInt("_EmissionCount", emissionCount); computeShader.SetVector("_Seeds", new Vector3(Random.value, Random.value, Random.value)); computeShader.SetBuffer(kernelEmit, "_AliveIndices_A", aliveIndicesBuffers[pingPongA - 1]); computeShader.SetInt("_PingPong_A", pingPongA); int emitThreadGroups = Mathf.CeilToInt((float)emissionCount / THREAD_COUNT_1D); computeShader.Dispatch(kernelEmit, emitThreadGroups, 1, 1); } } uint[] currentCounters = new uint[3]; countersBuffer.GetData(currentCounters); uint simulationCount = currentCounters[pingPongA]; if (simulationCount > 0) { computeShader.SetBuffer(kernelSimulate, "_AliveIndices_A", aliveIndicesBuffers[pingPongA - 1]); computeShader.SetBuffer(kernelSimulate, "_AliveIndices_B", aliveIndicesBuffers[pingPongB - 1]); computeShader.SetFloat("_DeltaTime", deltaTime); computeShader.SetInt("_PingPong_A", pingPongA); computeShader.SetInt("_PingPong_B", pingPongB); int simulateThreadGroups = Mathf.CeilToInt((float)simulationCount / THREAD_COUNT_1D); computeShader.Dispatch(kernelSimulate, simulateThreadGroups, 1, 1); } int temp = pingPongA; pingPongA = pingPongB; pingPongB = temp; }
    
    private void SetShaderParameters()
    {
        computeShader.SetFloat("_MinLifetime", minLifetime);
        computeShader.SetFloat("_MaxLifetime", maxLifetime);
        computeShader.SetFloat("_Drag", drag);
        computeShader.SetVector("_GlobalForce", globalForce);
        computeShader.SetBool("_CurlNoiseEnabled", curlNoiseEnabled);
        computeShader.SetFloat("_CurlNoiseScale", curlNoiseScale);
        computeShader.SetFloat("_CurlNoiseStrength", curlNoiseStrength);

        // 【新增】传递颜色控制参数
        computeShader.SetBool("_ColorOverLife", colorOverLife);
        computeShader.SetBool("_VelocityToColor", velocityToColor);
        computeShader.SetFloat("_MaxSpeedForColor", maxSpeedForColor);

        if (emitters.Count > 0) { var gpuEmitters = emitters.Select(e => new EmitterGPU { enabled = e.enabled ? 1 : 0, emissionRate = e.emissionRate, position = e.position, radius = e.radius, initialVelocity = e.initialVelocity, minInitialSpeed = e.minInitialSpeed, maxInitialSpeed = e.maxInitialSpeed, color = e.color }).ToArray(); emittersBuffer.SetData(gpuEmitters); }
        if (forceFields.Count > 0) { 
            // 强制禁用所有力场
            var gpuForceFields = forceFields.Select(f => new ForceFieldGPU { 
                enabled = 0, // 强制设置为0，禁用所有力场
                position = f.position, 
                radius = f.radius, 
                strength = 0.0f // 强制设置强度为0
            }).ToArray(); 
            forceFieldsBuffer.SetData(gpuForceFields); 
            Debug.Log($"🔧 强制禁用所有力场，数量: {forceFields.Count}");
        }
        computeShader.SetInt("_ForceFieldCount", forceFields.Count);
    }
    
    // InitializeBuffers, InitializeParticles, SetupCamera, ReleaseBuffers (无变化)
    void InitializeBuffers() { particleBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ParticleConstants.MAX_PARTICLES, sizeof(float) * 16); deadPoolBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ParticleConstants.MAX_PARTICLES, sizeof(uint)); aliveIndicesBufferA = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ParticleConstants.MAX_PARTICLES, sizeof(uint)); aliveIndicesBufferB = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ParticleConstants.MAX_PARTICLES, sizeof(uint)); aliveIndicesBuffers = new GraphicsBuffer[] { aliveIndicesBufferA, aliveIndicesBufferB }; countersBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 3, sizeof(uint)); indirectArgsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 1, 5 * sizeof(uint)); emittersBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Mathf.Max(1, emitters.Count), 60); forceFieldsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Mathf.Max(1, forceFields.Count), 24); }
    void InitializeParticles() { kernelUpdateArgs = computeShader.FindKernel("UpdateArgs"); kernelEmit = computeShader.FindKernel("Emit"); kernelSimulate = computeShader.FindKernel("Simulate"); uint[] deadIndices = new uint[ParticleConstants.MAX_PARTICLES]; for (uint i = 0; i < ParticleConstants.MAX_PARTICLES; i++) { deadIndices[i] = i; } deadPoolBuffer.SetData(deadIndices); uint[] counters = new uint[] { (uint)ParticleConstants.MAX_PARTICLES, 0, 0 }; countersBuffer.SetData(counters); uint[] args = new uint[5] { 0, 0, 0, 0, 0 }; if (particleMesh != null) args[0] = particleMesh.GetIndexCount(0); indirectArgsBuffer.SetData(args); }
    void SetupCamera() { 
        if (Camera.main != null) { 
            Camera.main.backgroundColor = Color.black; 
            Camera.main.clearFlags = CameraClearFlags.SolidColor; 
            Camera.main.transform.position = new Vector3(0, 5, -10); 
            Camera.main.transform.rotation = Quaternion.identity; 
            Camera.main.nearClipPlane = 0.1f;
            Camera.main.farClipPlane = 1000f;
        } 
    }
    void ReleaseBuffers() { particleBuffer?.Release(); deadPoolBuffer?.Release(); aliveIndicesBufferA?.Release(); aliveIndicesBufferB?.Release(); countersBuffer?.Release(); indirectArgsBuffer?.Release(); emittersBuffer?.Release(); forceFieldsBuffer?.Release(); }
}
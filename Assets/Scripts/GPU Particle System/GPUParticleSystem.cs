using UnityEngine;
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

    // =================================================================================
    //     DEBUGGING: 用于从GPU回读数据的结构体
    // =================================================================================
    [StructLayout(LayoutKind.Sequential)]
    private struct DebugData
    {
        public uint particleId;
        public Vector3 inPosition;
        public Vector3 attractorPosition;
        public float attractorStrength;
        public float attractorEnabled;
        public Vector3 acceleration;
        public Vector3 outPosition;
    }
    private GraphicsBuffer _debugBuffer;
    private DebugData[] _debugDataArray;
    private const int DEBUG_COUNT = 10;
    // =================================================================================

    [Header("Emitters")]
    public List<Emitter> emitters = new List<Emitter>();
    [Header("Global Forces")]
    public List<ForceField> forceFields = new List<ForceField>();
    public Vector3 globalForce = Vector3.zero;
    [Header("Curl Noise (Vortex) Settings")]
    public bool curlNoiseEnabled = true;
    public float curlNoiseScale = 0.05f;
    public float curlNoiseStrength = 5.0f;
    [Header("Simulation Settings")]
    public float minLifetime = 8.0f;
    public float maxLifetime = 12.0f;
    public float drag = 0.1f;
    [Header("Color Settings")]
    public bool colorOverLife = false;
    public bool velocityToColor = false;
    public float maxSpeedForColor = 20.0f;
    [Header("General Settings")]
    public bool prewarm = true;
    public float prewarmTime = 10.0f;
    public Vector3 renderBounds = new Vector3(100, 100, 100);
    [Header("References")]
    public ComputeShader computeShader;
    public Material particleMaterial;
    public Mesh particleMesh;

    private int kernelUpdateArgs, kernelEmit, kernelSimulate;
    private GraphicsBuffer particleBuffer, deadPoolBuffer, aliveIndicesBufferA, aliveIndicesBufferB, countersBuffer, indirectArgsBuffer;
    private GraphicsBuffer emittersBuffer, forceFieldsBuffer;
    private GraphicsBuffer[] aliveIndicesBuffers;
    private int pingPongA = 1, pingPongB = 2;
    private const int THREAD_COUNT_1D = 256;

    void OnEnable() { InitializeBuffers(); InitializeParticles(); SetupCamera(); if (prewarm) { Debug.Log("🔥 Pre-warming..."); float fixedDeltaTime = 1.0f / 30.0f; int prewarmSteps = Mathf.CeilToInt(prewarmTime / fixedDeltaTime); for (int i = 0; i < prewarmSteps; i++) { SetShaderParameters(); RunSimulationStep(fixedDeltaTime); } Debug.Log("🔥 Pre-warm complete."); } }
    void OnDisable() { ReleaseBuffers(); }

    void Update() 
    { 
        SetShaderParameters(); 
        RunSimulationStep(Time.deltaTime); 
        
        Bounds bounds = new Bounds(Vector3.zero, renderBounds); 
        particleMaterial.SetBuffer("_Particles", particleBuffer); 
        particleMaterial.SetBuffer("_AliveIndices", aliveIndicesBuffers[pingPongA - 1]); 
        Graphics.DrawMeshInstancedIndirect(particleMesh, 0, particleMaterial, bounds, indirectArgsBuffer); 
        
        // =================================================================================
        //     DEBUGGING: 在特定帧读取并打印GPU调试数据
        // =================================================================================
        if (Time.frameCount == 60)
        {
            _debugBuffer.GetData(_debugDataArray);
            Debug.LogWarning("--- GPU DEBUG PROBE (Frame 60) ---");
            for (int i = 0; i < DEBUG_COUNT; i++)
            {
                var data = _debugDataArray[i];
                string log = $"[Particle {i}] ID: {data.particleId}\n" +
                             $"  - In Position:       {data.inPosition}\n" +
                             $"  - Attractor State:   Pos={data.attractorPosition}, Str={data.attractorStrength}, Enabled={data.attractorEnabled}\n" +
                             $"  - Acceleration:      {data.acceleration}\n" +
                             $"  - Out Position:      {data.outPosition}";
                Debug.Log(log);
            }
            Debug.LogWarning("--- END GPU DEBUG PROBE ---");
        }
        // =================================================================================
    }
    
    private void SetShaderParameters()
    {
        computeShader.SetFloat("_MinLifetime", minLifetime);
        computeShader.SetFloat("_MaxLifetime", maxLifetime);
        computeShader.SetFloat("_Drag", drag);
        computeShader.SetVector("_GlobalForce", globalForce);
        computeShader.SetBool("_CurlNoiseEnabled", curlNoiseEnabled);
        computeShader.SetFloat("_CurlNoiseScale", curlNoiseScale);
        computeShader.SetFloat("_CurlNoiseStrength", curlNoiseStrength);
        computeShader.SetBool("_ColorOverLife", colorOverLife);
        computeShader.SetBool("_VelocityToColor", velocityToColor);
        computeShader.SetFloat("_MaxSpeedForColor", maxSpeedForColor);

        if (emitters.Count > 0) { var gpuEmitters = emitters.Select(e => new EmitterGPU { position = new Vector4(e.position.x, e.position.y, e.position.z, e.radius), initialVelocity = e.initialVelocity, speedMinMax = new Vector4(e.minInitialSpeed, e.maxInitialSpeed, 0, 0), color = e.color, ratesAndEnabled = new Vector4(e.emissionRate, e.enabled ? 1.0f : 0.0f, 0, 0) }).ToArray(); emittersBuffer.SetData(gpuEmitters); }
        if (forceFields.Count > 0) { var gpuForceFields = forceFields.Select(f => new ForceFieldGPU { positionAndRadius = new Vector4(f.position.x, f.position.y, f.position.z, f.radius), strengthAndEnabled = new Vector4(f.strength, f.enabled ? 1.0f : 0.0f, 0, 0) }).ToArray(); forceFieldsBuffer.SetData(gpuForceFields); }
        computeShader.SetInt("_ForceFieldCount", forceFields.Count);
    }

    private void RunSimulationStep(float deltaTime) 
    { 
        computeShader.SetBuffer(kernelEmit, "_Emitters", emittersBuffer); 
        computeShader.SetBuffer(kernelSimulate, "_ForceFields", forceFieldsBuffer); 
        computeShader.SetBuffer(kernelUpdateArgs, "_Counters", countersBuffer); 
        computeShader.SetBuffer(kernelUpdateArgs, "_IndirectArgs", indirectArgsBuffer); 
        computeShader.SetBuffer(kernelEmit, "_Particles", particleBuffer); 
        computeShader.SetBuffer(kernelEmit, "_DeadPool", deadPoolBuffer); 
        computeShader.SetBuffer(kernelEmit, "_Counters", countersBuffer); 
        computeShader.SetBuffer(kernelSimulate, "_Particles", particleBuffer); 
        computeShader.SetBuffer(kernelSimulate, "_DeadPool", deadPoolBuffer); 
        computeShader.SetBuffer(kernelSimulate, "_Counters", countersBuffer); 
        computeShader.SetBuffer(kernelSimulate, "_IndirectArgs", indirectArgsBuffer); 
        computeShader.SetInt("_PingPong_A", pingPongA); 
        computeShader.SetInt("_PingPong_B", pingPongB); 

        // =================================================================================
        //     DEBUGGING: 将调试缓冲区绑定到内核
        // =================================================================================
        computeShader.SetBuffer(kernelSimulate, "_DebugBuffer", _debugBuffer);
        // =================================================================================
        
        computeShader.Dispatch(kernelUpdateArgs, 1, 1, 1); 

        for (int i = 0; i < emitters.Count; i++) { if (!emitters[i].enabled) continue; int emissionCount = Mathf.RoundToInt(emitters[i].emissionRate * deltaTime); if (emissionCount > 0) { computeShader.SetInt("_EmitterIndex", i); computeShader.SetInt("_EmissionCount", emissionCount); computeShader.SetVector("_Seeds", new Vector3(Random.value, Random.value, Random.value)); computeShader.SetBuffer(kernelEmit, "_AliveIndices_A", aliveIndicesBuffers[pingPongA - 1]); computeShader.SetInt("_PingPong_A", pingPongA); int emitThreadGroups = Mathf.CeilToInt((float)emissionCount / THREAD_COUNT_1D); computeShader.Dispatch(kernelEmit, emitThreadGroups, 1, 1); } } 
        uint[] currentCounters = new uint[3]; countersBuffer.GetData(currentCounters); uint simulationCount = currentCounters[pingPongA]; if (simulationCount > 0) { computeShader.SetBuffer(kernelSimulate, "_AliveIndices_A", aliveIndicesBuffers[pingPongA - 1]); computeShader.SetBuffer(kernelSimulate, "_AliveIndices_B", aliveIndicesBuffers[pingPongB - 1]); computeShader.SetFloat("_DeltaTime", deltaTime); computeShader.SetInt("_PingPong_A", pingPongA); computeShader.SetInt("_PingPong_B", pingPongB); int simulateThreadGroups = Mathf.CeilToInt((float)simulationCount / THREAD_COUNT_1D); computeShader.Dispatch(kernelSimulate, simulateThreadGroups, 1, 1); } 
        int temp = pingPongA; pingPongA = pingPongB; pingPongB = temp; 
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
        emittersBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Mathf.Max(1, emitters.Count), Marshal.SizeOf<EmitterGPU>()); 
        forceFieldsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Mathf.Max(1, forceFields.Count), Marshal.SizeOf<ForceFieldGPU>()); 

        // =================================================================================
        //     DEBUGGING: 初始化调试缓冲区
        // =================================================================================
        _debugBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, DEBUG_COUNT, Marshal.SizeOf<DebugData>());
        _debugDataArray = new DebugData[DEBUG_COUNT];
        // =================================================================================
    }

    void InitializeParticles() { kernelUpdateArgs = computeShader.FindKernel("UpdateArgs"); kernelEmit = computeShader.FindKernel("Emit"); kernelSimulate = computeShader.FindKernel("Simulate"); uint[] deadIndices = new uint[ParticleConstants.MAX_PARTICLES]; for (uint i = 0; i < ParticleConstants.MAX_PARTICLES; i++) { deadIndices[i] = i; } deadPoolBuffer.SetData(deadIndices); uint[] counters = new uint[] { (uint)ParticleConstants.MAX_PARTICLES, 0, 0 }; countersBuffer.SetData(counters); uint[] args = new uint[5] { 0, 0, 0, 0, 0 }; if (particleMesh != null) args[0] = particleMesh.GetIndexCount(0); indirectArgsBuffer.SetData(args); }
    void SetupCamera() { if (Camera.main != null) { Camera.main.backgroundColor = Color.black; Camera.main.clearFlags = CameraClearFlags.SolidColor; } }

    void ReleaseBuffers() 
    { 
        particleBuffer?.Release(); deadPoolBuffer?.Release(); aliveIndicesBufferA?.Release(); aliveIndicesBufferB?.Release(); 
        countersBuffer?.Release(); indirectArgsBuffer?.Release(); emittersBuffer?.Release(); forceFieldsBuffer?.Release(); 
        // =================================================================================
        //     DEBUGGING: 释放调试缓冲区
        // =================================================================================
        _debugBuffer?.Release();
        // =================================================================================
    }
}
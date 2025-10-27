using UnityEngine;
using UnityEngine.Rendering.Universal;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Runtime.InteropServices;

public class GPUParticleSystem : MonoBehaviour
{
	[System.Serializable]
	public struct Emitter { public string name; public bool enabled; public int emissionRate; public Vector3 position; public float radius; public Vector3 initialVelocity; public float minInitialSpeed; public float maxInitialSpeed; public Color color; }
	
    // Old force-field code removed

    // Legacy vortex/ribbon modes removed

	[StructLayout(LayoutKind.Sequential)]
	private struct EmitterGPU { public Vector4 position; public Vector4 initialVelocity; public Vector4 speedMinMax; public Color color; public Vector4 ratesAndEnabled; }

	//

    // No extra structs required

    // Debug system removed
    private int _lastVisiblePingPong = 1; // visible alive list used by renderer

    // Emitters removed; use global emission rate
	[Header("Global Emission")]
	public int globalEmissionRate = 2000;
	public Color baseColor = Color.white;

	[Header("Burst Emission")]
	public bool burstEnabled = true;
    public float burstInterval = 5f; // every 5 seconds
    public int burstParticleCount = 20000; // extra particles per burst
	private float _burstTimer = 0f;

	[Header("Mouse Attractor")] 
	public bool mouseAttractorEnabled = true;
    public float mouseAttractorStrength = 200f; // base attraction strength
    public float mouseAttractorFalloff = 1.0f; // 1/r^falloff
    public float mouseAttractorDragBoost = 0.0f; // local damping near mouse
    [Tooltip("Extra attraction multiplier for mouse-emitted particles")] public float mouseAttractorStrengthMouseFactor = 2.5f;
    [Tooltip("If true, mouse-emitted particles ignore center gravity")] public bool mouseParticlesIgnoreCenterGravity = true;
    public int mouseEmissionRate = 500; // extra particles per second from mouse
	public float mouseEmissionSpeed = 6f;
    [Tooltip("Initial random jitter speed for mouse-emitted particles")] public float mouseEmitJitterSpeed = 0.3f;
    [Tooltip("Extra damping for mouse-emitted particles (added to drag)")] public float mouseBornExtraDamping = 2.0f;
    [Tooltip("Speed clamp for mouse-emitted particles")] public float mouseBornMaxSpeed = 2.0f;
    [Tooltip("Allow regular particles to break shell constraints when attracted to mouse")] public bool allowShellBreakToMouse = true;
    public float mousePlaneY = 0f; // projection plane height for mouse
	private Vector3 _mouseWorldPos;
	private bool _mouseActive;

    // Ring feature removed

	[Header("Global Forces")]
	public Vector3 globalForce = Vector3.zero;
	
    // Galactic disk mode and colors removed

	[Header("Spherical Mode")] 
	public Vector3 sphereCenter = Vector3.zero;
	public float sphereRadius = 50f;
	[Range(0f, 50f)] public float shellThickness = 2f;
	public enum SphereVelocityMode { RadialOut = 0, RadialIn = 1, Tangential = 2, RadialPlusSwirl = 3 }
	public SphereVelocityMode sphereVelocityMode = SphereVelocityMode.Tangential;
	[Header("Spherical Velocities")]
	public float radialSpeed = 1f;
	public float tangentialSpeed = 5f;
	public float swirlSpeed = 0f;
    [Header("Spherical Dynamics")]
    public float gravityK = 8f; // attraction to center (~1/r^2)
    [Range(0f, 1f)] public float radialDamping = 0.3f; // radial velocity damping
	public enum SphereBoundaryMode { None = 0, SurfaceStick = 1, ShellClamp = 2 }
	public SphereBoundaryMode sphereBoundaryMode = SphereBoundaryMode.SurfaceStick;
    [Tooltip("Spring strength back to target radius")]
	public float boundarySpring = 12f;
    [Tooltip("Velocity damping for the spring constraint")]
	public float boundaryDamping = 2.2f;

    [Header("Shell Emission Pulse")] 
	public bool shellEmissionPulseEnabled = true;
	public int shellEmissionMin = 1000;
	public int shellEmissionMax = 3000;
    [Tooltip("Half-period in seconds. 3 means 0-3s up, 3-6s down, loop.")]
	public float shellEmissionHalfPeriod = 3f;
	private float _shellEmissionTimer = 0f;
	[Header("Spherical Colors")]
	public bool sphereColorByRadiusEnabled = true;
	public Color sphereInnerColor = new Color(0.95f, 0.98f, 1.0f);
	public Color sphereMidColor = new Color(0.7f, 0.85f, 1.0f);
	public Color sphereOuterColor = new Color(0.35f, 0.55f, 1.0f);
    [Header("Temperature Color")] 
	public bool temperatureColorEnabled = true;
    [Range(20f,30f)] public float temperatureC = 25f; // 20~30 Celsius
	public Color tempColorLow = new Color(0.7f, 0.85f, 1.0f);
	public Color tempColorHigh = new Color(1.0f, 0.6f, 0.3f);
	
	[Header("Simulation Settings")]
	public float minLifetime = 8.0f;
	public float maxLifetime = 12.0f;
	public float drag = 0.1f;
	[Header("Color Settings")]
    public bool colorOverLife = false; // keep emitter color unless overridden
    public bool velocityToColor = true; // map velocity to color
	public float maxSpeedForColor = 20.0f;
	[Header("General Settings")]
     public bool prewarm = true;   // prewarm so particles exist on start
	public float prewarmTime = 10.0f;
	public Vector3 renderBounds = new Vector3(100, 100, 100);
	[Header("References")]
	public ComputeShader computeShader;
	public Material particleMaterial;
	public Mesh particleMesh;
	[Header("Curves (1D Textures)")]
	public Texture2D colorOverLifeTex;
	public Texture2D sizeOverLifeTex;
    [Tooltip("Generate and save default 1D curve textures to Assets/Textures/Curves")] public bool autoGenerateCurvesIfMissing = true;

	[Header("Camera Overrides (Runtime)")]
	public bool forceBlackBackground = true;
	public Color backgroundColor = Color.black;
	public bool disablePostProcessing = true;
	public bool autoFrameCameraOnStart = true;
    public float cameraElevationDeg = 25f;   // camera elevation in degrees
    public float cameraYawDeg = 30f;         // camera yaw in degrees
    public float cameraMargin = 1.15f;       // screen margin factor

	// 旧 Ring/Flow 参数组已移除

	private int kernelUpdateArgs, kernelEmit, kernelSimulate;
	private GraphicsBuffer particleBuffer, deadPoolBuffer, aliveIndicesBufferA, aliveIndicesBufferB, countersBuffer, indirectArgsBuffer;
	private GraphicsBuffer globalIdCounterBuffer;  // 全局ID计数器缓冲区
	private GraphicsBuffer[] aliveIndicesBuffers;
	private int pingPongA = 1, pingPongB = 2;
	private const int THREAD_COUNT_1D = 256;

    void OnEnable() 
	{ 
        if (autoGenerateCurvesIfMissing && (colorOverLifeTex == null || sizeOverLifeTex == null))
        {
            GenerateCurve1DTextures();
        }
		InitializeBuffers(); 
		InitializeParticles(); 
		SetupCamera(); 
        // no default random config
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
    // Shell pulse emission (sphere shell only)
	if (shellEmissionPulseEnabled)
	{
		_shellEmissionTimer += Time.deltaTime; 
		float T = Mathf.Max(0.001f, shellEmissionHalfPeriod * 2f); // 一个完整周期 
		float t = _shellEmissionTimer % T; 
		float half = T * 0.5f; 
		float phase01 = t < half ? (t / half) : (1f - (t - half) / half); // 0->1->0 三角波 
		globalEmissionRate = Mathf.RoundToInt(Mathf.Lerp(shellEmissionMin, shellEmissionMax, phase01));
	}

    // Mouse world position via screen ray -> plane y=mousePlaneY
	if (mouseAttractorEnabled)
	{
		var cam = Camera.main;
		if (cam != null)
		{
			Ray ray = cam.ScreenPointToRay(Input.mousePosition);
			Plane plane = new Plane(Vector3.up, new Vector3(0, mousePlaneY, 0));
			float enter;
			if (plane.Raycast(ray, out enter))
			{
				_mouseWorldPos = ray.GetPoint(enter);
                _mouseActive = Input.mousePresent; // pointer device present
			}
			else { _mouseActive = false; }
		}
		else { _mouseActive = false; }
	}

	SetShaderParameters(); 
	
	var didSimulate = RunSimulationStep(Time.deltaTime);
	
	
	Bounds bounds = new Bounds(Vector3.zero, renderBounds); 
	particleMaterial.SetBuffer("_Particles", particleBuffer); 
    // Use the alive list matching the last simulation pass
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
        // Legacy curl/multi/ring/flow removed
		computeShader.SetBool("_ColorOverLife", colorOverLife);
		computeShader.SetBool("_VelocityToColor", velocityToColor);
		computeShader.SetFloat("_MaxSpeedForColor", maxSpeedForColor);

        // Spherical uniforms (shell-only system)
		computeShader.SetVector("_SphereCenter", sphereCenter);
		computeShader.SetFloat("_SphereRadius", sphereRadius);
		computeShader.SetFloat("_ShellThickness", shellThickness);
		computeShader.SetInt("_SphereVelocityMode", (int)sphereVelocityMode);
		computeShader.SetFloat("_RadialSpeed", radialSpeed);
		computeShader.SetFloat("_TangentialSpeed", tangentialSpeed);
		computeShader.SetFloat("_SwirlSpeed", swirlSpeed);
		computeShader.SetFloat("_GravityK", gravityK);
		computeShader.SetFloat("_RadialDamping", radialDamping);
		computeShader.SetInt("_SphereBoundaryMode", (int)sphereBoundaryMode);
		computeShader.SetFloat("_BoundarySpring", boundarySpring);
		computeShader.SetFloat("_BoundaryDamping", boundaryDamping);
		computeShader.SetBool("_SphereColorByRadiusEnabled", sphereColorByRadiusEnabled);
		computeShader.SetVector("_SphereInnerColor", (Vector4)sphereInnerColor);
		computeShader.SetVector("_SphereMidColor", (Vector4)sphereMidColor);
		computeShader.SetVector("_SphereOuterColor", (Vector4)sphereOuterColor);
		computeShader.SetVector("_BaseColor", (Vector4)baseColor);

		// Temperature uniforms
		computeShader.SetBool("_TemperatureColorEnabled", temperatureColorEnabled);
		computeShader.SetFloat("_TemperatureC", temperatureC);
		computeShader.SetVector("_TempColorLow", (Vector4)tempColorLow);
		computeShader.SetVector("_TempColorHigh", (Vector4)tempColorHigh);

		// Mouse Attractor
		computeShader.SetBool("_MouseAttractorEnabled", mouseAttractorEnabled && _mouseActive);
		computeShader.SetVector("_MouseWorldPos", _mouseWorldPos);
		computeShader.SetFloat("_MouseAttractorStrength", mouseAttractorStrength);
		computeShader.SetFloat("_MouseAttractorFalloff", mouseAttractorFalloff);
		computeShader.SetFloat("_MouseAttractorDragBoost", mouseAttractorDragBoost);
		computeShader.SetFloat("_MouseAttractorStrengthMouseFactor", mouseAttractorStrengthMouseFactor);
		computeShader.SetBool("_MouseParticlesIgnoreCenterGravity", mouseParticlesIgnoreCenterGravity);
		computeShader.SetFloat("_MouseEmitJitterSpeed", mouseEmitJitterSpeed);
		computeShader.SetFloat("_MouseBornExtraDamping", mouseBornExtraDamping);
		computeShader.SetFloat("_MouseBornMaxSpeed", mouseBornMaxSpeed);
		computeShader.SetBool("_AllowShellBreakToMouse", allowShellBreakToMouse);

        // Ring uniforms removed
	}
	
	bool RunSimulationStep(float deltaTime) 
	{ 
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
		
        // Regular emission: global rate
		int emissionCount = Mathf.RoundToInt(Mathf.Max(0, globalEmissionRate) * deltaTime);
		if (emissionCount > 0)
		{
			computeShader.SetInt("_EmissionCount", emissionCount); 
			computeShader.SetVector("_Seeds", new Vector3(Random.value, Random.value, Random.value)); 
			computeShader.SetBuffer(kernelEmit, "_AliveIndices_A", aliveIndicesBuffers[pingPongA - 1]); 
			computeShader.SetInt("_PingPong_A", pingPongA); 
			int emitThreadGroups = Mathf.CeilToInt((float)emissionCount / THREAD_COUNT_1D); 
			computeShader.Dispatch(kernelEmit, emitThreadGroups, 1, 1); 
		}

        // Timed burst emission
		if (burstEnabled && burstParticleCount > 0)
		{
			_burstTimer += deltaTime;
			if (_burstTimer >= Mathf.Max(0.1f, burstInterval))
			{
				_burstTimer = 0f;
				int burstCount = burstParticleCount;
				computeShader.SetInt("_EmissionCount", burstCount); 
				computeShader.SetVector("_Seeds", new Vector3(Random.value, Random.value, Random.value)); 
				computeShader.SetBuffer(kernelEmit, "_AliveIndices_A", aliveIndicesBuffers[pingPongA - 1]); 
				computeShader.SetInt("_PingPong_A", pingPongA); 
				int burstGroups = Mathf.CeilToInt((float)burstCount / THREAD_COUNT_1D); 
				computeShader.Dispatch(kernelEmit, burstGroups, 1, 1); 
			}
		}

        // Mouse extra emission
		if (mouseAttractorEnabled && _mouseActive && mouseEmissionRate > 0)
		{
			int mouseCount = Mathf.RoundToInt(mouseEmissionRate * deltaTime);
			if (mouseCount > 0)
			{
				computeShader.SetInt("_EmissionCount", mouseCount); 
				computeShader.SetVector("_Seeds", new Vector3(Random.value, Random.value, Random.value)); 
				computeShader.SetBuffer(kernelEmit, "_AliveIndices_A", aliveIndicesBuffers[pingPongA - 1]); 
				computeShader.SetInt("_PingPong_A", pingPongA); 
				// 标记利用 attributes.w 在 Emit 中区分鼠标发射（由 compute 决定）
				computeShader.SetBool("_EmitFromMouse", true);
				computeShader.SetFloat("_MouseEmissionSpeed", mouseEmissionSpeed);
				int emitThreadGroups2 = Mathf.CeilToInt((float)mouseCount / THREAD_COUNT_1D); 
				computeShader.Dispatch(kernelEmit, emitThreadGroups2, 1, 1); 
				computeShader.SetBool("_EmitFromMouse", false);
			}
		}
		
        // Read current counters
		uint[] currentCounters = new uint[3]; 
		countersBuffer.GetData(currentCounters); 
		uint simulationCount = currentCounters[pingPongA]; 
		bool simulateExecuted = false;
		
        // Run simulation if there are alive particles
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
        globalIdCounterBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 2, sizeof(uint));  // global id counter [0]=particle id, [1]=debug idx
        // emitter buffer removed
	}

	void InitializeParticles() 
	{ 
		kernelUpdateArgs = computeShader.FindKernel("UpdateArgs"); 
		kernelEmit = computeShader.FindKernel("Emit"); 
		kernelSimulate = computeShader.FindKernel("Simulate"); 
		Particle[] initialParticles = new Particle[ParticleConstants.MAX_PARTICLES];
        // Fill defaults (optional but good hygiene)
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
        // Upload clean array to GPU buffers
		particleBuffer.SetData(initialParticles);
		uint[] deadIndices = new uint[ParticleConstants.MAX_PARTICLES]; 
		for (uint i = 0; i < ParticleConstants.MAX_PARTICLES; i++) 
		{ 
			deadIndices[i] = i; 
		} 
		deadPoolBuffer.SetData(deadIndices); 
		uint[] counters = new uint[] { (uint)ParticleConstants.MAX_PARTICLES, 0, 0 }; 
		countersBuffer.SetData(counters); 
        uint[] globalIdCounter = new uint[] { 0, 0 };  // [0]=particle id, [1]=debug idx
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
			FrameCameraToSphere(cam);
		}
	}

	[ContextMenu("Frame Camera To Sphere")]
	void CM_FrameCameraSphere() { var cam = Camera.main; if (cam!=null) FrameCameraToSphere(cam); }

	void FrameCameraToSphere(Camera cam)
	{
        Vector3 center = sphereCenter;
		float R = Mathf.Max(1f, sphereRadius * 1.05f);
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
        // emittersBuffer removed
        // debug buffers removed
	}
	
    // Debug helpers removed

    [ContextMenu("Curves/Generate 1D Textures")] 
    void GenerateCurve1DTextures()
    {
        int width = 256;
        // Color over life (0..1)
        Texture2D colTex = new Texture2D(width, 1, TextureFormat.RGBA32, false, true);
        colTex.wrapMode = TextureWrapMode.Clamp;
        colTex.filterMode = FilterMode.Bilinear;
        for (int x = 0; x < width; x++)
        {
            float t = x / (float)(width - 1);
            Color c = Color.Lerp(tempColorLow, tempColorHigh, t);
            float a = Mathf.SmoothStep(0f, 1f, Mathf.Sin(t * Mathf.PI));
            c.a = a;
            colTex.SetPixel(x, 0, c);
        }
        colTex.Apply(false, false);

        // Size over life
        Texture2D sizeTex = new Texture2D(width, 1, TextureFormat.R8, false, true);
        sizeTex.wrapMode = TextureWrapMode.Clamp;
        sizeTex.filterMode = FilterMode.Bilinear;
        for (int x = 0; x < width; x++)
        {
            float t = x / (float)(width - 1);
            float s = Mathf.Sin(t * Mathf.PI);
            float val = Mathf.Lerp(0.25f, 1.0f, s);
            Color r = new Color(val, val, val, 1f);
            sizeTex.SetPixel(x, 0, r);
        }
        sizeTex.Apply(false, false);

        colorOverLifeTex = colTex;
        sizeOverLifeTex = sizeTex;

        try
        {
            string dir = Path.Combine(Application.dataPath, "Textures/Curves");
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
            string colPath = Path.Combine(dir, "ColorOverLife.png");
            string sizePath = Path.Combine(dir, "SizeOverLife.png");
            File.WriteAllBytes(colPath, colTex.EncodeToPNG());
            File.WriteAllBytes(sizePath, sizeTex.EncodeToPNG());
#if UNITY_EDITOR
            UnityEditor.AssetDatabase.Refresh();
#endif
        }
        catch {}
    }

    // Ring quick setup removed
}

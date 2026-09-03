/**
 * 3D Model Viewer and Measurement Visualizer using Three.js
 */

let scene, camera, renderer, controls;
let currentMesh = null;
let slicePlanesGroup = new THREE.Group();
let gridHelper, axesHelper;

function init3DViewer() {
  const container = document.getElementById('viewport3d');
  if (!container) return;

  const width = container.clientWidth;
  const height = container.clientHeight;

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0e17);
  scene.fog = new THREE.FogExp2(0x0a0e17, 0.0015);

  // Camera
  camera = new THREE.PerspectiveCamera(45, width / height, 1, 3000);
  camera.position.set(200, 250, 300);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  // Orbit Controls
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxPolarAngle = Math.PI / 2 + 0.1; // Don't go below floor

  // Lights
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambientLight);

  const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight1.position.set(150, 300, 200);
  dirLight1.castShadow = true;
  scene.add(dirLight1);

  const dirLight2 = new THREE.DirectionalLight(0x6366f1, 0.4);
  dirLight2.position.set(-150, -100, -150);
  scene.add(dirLight2);

  // Metric Floor Grid (10mm spacing, 300mm size)
  gridHelper = new THREE.GridHelper(400, 40, 0x6366f1, 0x1f293d);
  gridHelper.position.y = 0;
  scene.add(gridHelper);

  // Axes Helper (X: Red, Y: Green, Z: Blue)
  axesHelper = new THREE.AxesHelper(60);
  scene.add(axesHelper);

  scene.add(slicePlanesGroup);

  // Handle Resize
  window.addEventListener('resize', onWindowResize);

  // Animation loop
  animate();
}

function onWindowResize() {
  const container = document.getElementById('viewport3d');
  if (!container || !renderer || !camera) return;
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
  requestAnimationFrame(animate);
  if (controls) controls.update();
  if (renderer && scene && camera) renderer.render(scene, camera);
}

// Load STL Model
function loadSTLModel(stlUrl, measurements) {
  const loader = new THREE.STLLoader();
  const statusEl = document.getElementById('viewer-status');
  if (statusEl) statusEl.innerText = 'Cargando modelo STL...';

  loader.load(
    stlUrl,
    function (geometry) {
      if (currentMesh) scene.remove(currentMesh);
      geometry.computeVertexNormals();
      geometry.computeBoundingBox();

      const material = new THREE.MeshStandardMaterial({
        color: 0x38bdf8,
        metalness: 0.15,
        roughness: 0.4,
        side: THREE.DoubleSide
      });

      currentMesh = new THREE.Mesh(geometry, material);
      currentMesh.castShadow = true;
      currentMesh.receiveShadow = true;

      // Z-up to Three.js Y-up conversion
      currentMesh.rotation.x = -Math.PI / 2;

      scene.add(currentMesh);

      // Adjust camera to fit bounding box
      const bbox = geometry.boundingBox;
      const size = new THREE.Vector3();
      bbox.getSize(size);
      const maxDim = Math.max(size.x, size.y, size.z);

      camera.position.set(maxDim * 1.5, maxDim * 1.2, maxDim * 1.8);
      controls.target.set(0, size.z / 2, 0);
      controls.update();

      // Render measurement slices
      renderSlices(measurements);

      if (statusEl) {
        statusEl.innerText = `Modelo cargado: ${size.x.toFixed(1)} x ${size.y.toFixed(1)} x ${size.z.toFixed(1)} mm`;
      }
    },
    function (xhr) {
      if (statusEl && xhr.total > 0) {
        const pct = Math.round((xhr.loaded / xhr.total) * 100);
        statusEl.innerText = `Descargando 3D: ${pct}%`;
      }
    },
    function (error) {
      console.error('Error loading STL:', error);
      if (statusEl) statusEl.innerText = 'Error al cargar STL.';
    }
  );
}

// Draw Slicing Rings
function renderSlices(measurements) {
  // Clear previous slices
  while (slicePlanesGroup.children.length > 0) {
    slicePlanesGroup.remove(slicePlanesGroup.children[0]);
  }

  if (!measurements || !measurements.slices) return;

  const sliceMaterial = new THREE.LineBasicMaterial({
    color: 0xf59e0b,
    linewidth: 2,
    transparent: true,
    opacity: 0.8
  });

  measurements.slices.forEach(slice => {
    const radius = slice.equivalent_diameter_mm / 2.0;
    const heightY = slice.height_z_mm; // Map Z to Y

    const circleGeo = new THREE.BufferGeometry();
    const points = [];
    const segments = 48;
    for (let i = 0; i <= segments; i++) {
      const theta = (i / segments) * Math.PI * 2;
      points.push(new THREE.Vector3(Math.cos(theta) * radius, heightY, Math.sin(theta) * radius));
    }
    circleGeo.setFromPoints(points);

    const line = new THREE.Line(circleGeo, sliceMaterial);
    slicePlanesGroup.add(line);
  });
}

// UI Toggles
function toggleWireframe() {
  if (!currentMesh) return;
  currentMesh.material.wireframe = !currentMesh.material.wireframe;
}

function toggleSlices() {
  slicePlanesGroup.visible = !slicePlanesGroup.visible;
}

function resetView() {
  if (!camera || !controls) return;
  camera.position.set(200, 250, 300);
  controls.target.set(0, 50, 0);
  controls.update();
}

window.onload = function () {
  init3DViewer();
};

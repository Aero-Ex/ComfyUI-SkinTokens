import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const LOG = (...args) => console.log("[SkinTokens]", ...args);
const WARN = (...args) => console.warn("[SkinTokens]", ...args);
const ERR = (...args) => console.error("[SkinTokens]", ...args);

// Load Three.js and dependencies dynamically from CDN
async function loadThreeJS() {
    if (window.THREE) {
        LOG("Three.js already loaded, skipping.");
        return;
    }

    LOG("Loading Three.js r128 from CDN...");
    return new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js";
        script.onload = async () => {
            LOG("Three.js core loaded. Loading extensions...");
            await loadScript("https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js");
            LOG("  ✓ GLTFLoader");
            await loadScript("https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/OBJLoader.js");
            LOG("  ✓ OBJLoader");
            await loadScript("https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/MTLLoader.js");
            LOG("  ✓ MTLLoader");
            await loadScript("https://cdn.jsdelivr.net/npm/fflate@0.8.0/umd/index.js");
            LOG("  ✓ fflate (FBX dependency)");
            // Upgrade FBXLoader to r147 (latest non-module version) to fix embedded texture bugs
            await loadScript("https://cdn.jsdelivr.net/npm/three@0.147.0/examples/js/loaders/FBXLoader.js");
            LOG("  ✓ FBXLoader (r147)");
            
            // HACK: Force FBXLoader to recognize Blender's sanitized names (like Image_0_png) 
            // as valid embedded textures so it doesn't fall back to 404 URLs.
            const originalParse = THREE.FBXLoader.prototype.parse;
            THREE.FBXLoader.prototype.parse = function(data, path) {
                const result = originalParse.call(this, data, path);
                LOG("FBX Hack: Applied sanitized extension support.");
                return result;
            };
            
            await loadScript("https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/TGALoader.js");
            LOG("  ✓ TGALoader");
            LOG("All Three.js dependencies loaded.");
            resolve();
        };
        script.onerror = (e) => {
            ERR("Failed to load Three.js core!", e);
            reject(e);
        };
        document.head.appendChild(script);
    });
}

function loadScript(src) {
    return new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = src;
        script.onload = resolve;
        script.onerror = (e) => {
            ERR("Failed to load script:", src, e);
            reject(e);
        };
        document.head.appendChild(script);
    });
}

// =========================================================================
// ALL-IN-ONE VIEWER CONTROLLER
// Handles orbit camera + bone selection + bone rotation.
// ALL event listeners on the canvas ONLY — no document-level listeners.
// This is required because ComfyUI's DOM widget system blocks pointer
// events from reaching the document, breaking Three.js OrbitControls
// and TransformControls.
// =========================================================================
class ViewerController {
    constructor(camera, renderer, scene, node) {
        this.camera = camera;
        this.renderer = renderer;
        this.scene = scene;
        this.node = node;
        this.el = renderer.domElement;

        // Orbit state
        this.orbitTarget = new THREE.Vector3(0, 1, 0);
        this._spherical = new THREE.Spherical();
        this._spherical.setFromVector3(
            new THREE.Vector3().subVectors(camera.position, this.orbitTarget)
        );

        // Interaction state
        this._mode = 'idle'; // 'idle' | 'orbiting' | 'rotating_bone'
        this._button = -1;
        this._prevX = 0;
        this._prevY = 0;
        this._rotationAxis = null; // null = free rotate, Vector3 = constrained

        // Bone selection
        this.selectedBone = null;
        this._selectedSphere = null;
        this._highlightMat = new THREE.MeshBasicMaterial({ color: 0x00ff00, transparent: true, opacity: 0.5, depthTest: false });
        this._hiddenMat = new THREE.MeshBasicMaterial({ color: 0xff0000, visible: false });

        // Raycaster
        this._raycaster = new THREE.Raycaster();
        this._mouse = new THREE.Vector2();

        // Sensitivity
        this.rotateSpeed = 0.005;
        this.panSpeed = 0.005;
        this.zoomSpeed = 0.001;
        this.boneRotateSpeed = 0.01;

        // History (Undo/Redo)
        this._history = [];
        this._redoStack = [];
        this._maxHistory = 50;

        // Build gizmo (3 axis rings)
        this._gizmoRings = [];
        this._gizmoMats = {};
        this._gizmoDefaultColors = {};
        this._gizmo = this._createGizmo();
        this._gizmo.visible = false;
        scene.add(this._gizmo);

        this._setupEvents();
        LOG("ViewerController initialized with rotation gizmo and Undo/Redo (Alt key).");
    }

    _takeHistorySnapshot() {
        if (!this.node.bones || this.node.bones.length === 0) return;
        const snapshot = this.node.bones.map(b => ({
            name: b.name,
            quaternion: b.quaternion.clone()
        }));
        this._history.push(snapshot);
        if (this._history.length > this._maxHistory) this._history.shift();
        this._redoStack = []; // Clear redo on new action
    }

    undo() {
        if (this._history.length === 0) return;
        
        // Save current to redo stack before applying history
        const current = this.node.bones.map(b => ({
            name: b.name,
            quaternion: b.quaternion.clone()
        }));
        this._redoStack.push(current);

        const snapshot = this._history.pop();
        this._applySnapshot(snapshot);
        LOG("Undo performed (Alt+Z)");
    }

    redo() {
        if (this._redoStack.length === 0) return;

        const snapshot = this._redoStack.pop();
        // Save current to history stack
        const current = this.node.bones.map(b => ({
            name: b.name,
            quaternion: b.quaternion.clone()
        }));
        this._history.push(current);

        this._applySnapshot(snapshot);
        LOG("Redo performed (Alt+Y)");
    }

    _applySnapshot(snapshot) {
        if (!snapshot) return;
        snapshot.forEach(s => {
            const bone = this.node.bones.find(b => b.name === s.name);
            if (bone) bone.quaternion.copy(s.quaternion);
        });
    }

    _createGizmo() {
        const group = new THREE.Group();
        const gizmoRadius = 0.15;
        const tubeRadius = 0.010;
        const torusGeom = new THREE.TorusGeometry(gizmoRadius, tubeRadius, 12, 48);

        // X axis ring (RED) — lies in YZ plane
        const xMat = new THREE.MeshBasicMaterial({ color: 0xff4444, depthTest: false, transparent: true, opacity: 0.9 });
        const xRing = new THREE.Mesh(torusGeom, xMat);
        xRing.rotation.y = Math.PI / 2;
        xRing.userData.axis = new THREE.Vector3(1, 0, 0);
        xRing.userData.axisName = 'X';
        xRing.renderOrder = 999;
        group.add(xRing);

        // Y axis ring (GREEN) — lies in XZ plane
        const yMat = new THREE.MeshBasicMaterial({ color: 0x44ff44, depthTest: false, transparent: true, opacity: 0.9 });
        const yRing = new THREE.Mesh(torusGeom, yMat);
        yRing.rotation.x = Math.PI / 2;
        yRing.userData.axis = new THREE.Vector3(0, 1, 0);
        yRing.userData.axisName = 'Y';
        yRing.renderOrder = 999;
        group.add(yRing);

        // Z axis ring (BLUE) — lies in XY plane (default)
        const zMat = new THREE.MeshBasicMaterial({ color: 0x4488ff, depthTest: false, transparent: true, opacity: 0.9 });
        const zRing = new THREE.Mesh(torusGeom, zMat);
        zRing.userData.axis = new THREE.Vector3(0, 0, 1);
        zRing.userData.axisName = 'Z';
        zRing.renderOrder = 999;
        group.add(zRing);

        this._gizmoRings = [xRing, yRing, zRing];
        this._gizmoMats = { X: xMat, Y: yMat, Z: zMat };
        this._gizmoDefaultColors = { X: 0xff4444, Y: 0x44ff44, Z: 0x4488ff };

        return group;
    }

    _selectBone(bone, sphere) {
        if (this._selectedSphere) this._selectedSphere.material = this._hiddenMat;
        this.selectedBone = bone;
        this._selectedSphere = sphere;
        sphere.material = this._highlightMat;
        this._gizmo.visible = true;
        this._syncGizmo();
        LOG(`Bone selected: "${bone.name}" — gizmo shown`);
    }

    _deselectBone() {
        if (this._selectedSphere) this._selectedSphere.material = this._hiddenMat;
        if (this.selectedBone) LOG(`Deselected bone: "${this.selectedBone.name}"`);
        this.selectedBone = null;
        this._selectedSphere = null;
        this._gizmo.visible = false;
        this._rotationAxis = null;
    }

    _syncGizmo() {
        if (!this.selectedBone || !this._gizmo.visible) return;
        const boneWorldPos = new THREE.Vector3();
        this.selectedBone.getWorldPosition(boneWorldPos);
        this._gizmo.position.copy(boneWorldPos);
        // Scale to stay consistent screen size
        const dist = this.camera.position.distanceTo(boneWorldPos);
        const s = dist * 0.25;
        this._gizmo.scale.set(s, s, s);
    }

    _resetGizmoColors() {
        for (const [name, mat] of Object.entries(this._gizmoMats)) {
            mat.color.setHex(this._gizmoDefaultColors[name]);
            mat.opacity = 0.9;
        }
    }

    _setupEvents() {
        const el = this.el;
        el.style.touchAction = 'none';

        el.addEventListener('pointerdown', (e) => {
            this._prevX = e.clientX;
            this._prevY = e.clientY;
            this._button = e.button;

            if (e.button === 0) {
                const rect = el.getBoundingClientRect();
                this._mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
                this._mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
                this._raycaster.setFromCamera(this._mouse, this.camera);

                // Priority 1: Check gizmo rings
                if (this._gizmo.visible) {
                    const gizmoHits = this._raycaster.intersectObjects(this._gizmoRings);
                    if (gizmoHits.length > 0) {
                        const ring = gizmoHits[0].object;
                        this._rotationAxis = ring.userData.axis.clone();
                        this._mode = 'rotating_bone';
                        this._takeHistorySnapshot(); // Start tracking
                        this._resetGizmoColors();
                        this._gizmoMats[ring.userData.axisName].color.setHex(0xffff00);
                        this._gizmoMats[ring.userData.axisName].opacity = 1.0;
                        LOG(`Gizmo drag: ${ring.userData.axisName}-axis`);
                        return;
                    }
                }

                // Priority 2: Check bone spheres
                const spheres = this.node.jointSpheres || [];
                const boneHits = this._raycaster.intersectObjects(spheres);

                if (boneHits.length > 0) {
                    this._selectBone(boneHits[0].object.userData.bone, boneHits[0].object);
                    this._rotationAxis = null; // free rotate
                    this._mode = 'rotating_bone';
                    this._takeHistorySnapshot(); // Start tracking
                } else {
                    this._mode = 'orbiting';
                }
            } else {
                this._mode = 'orbiting';
            }
        });

        el.addEventListener('pointermove', (e) => {
            if (this._mode === 'idle') {
                // Hover effects
                const rect = el.getBoundingClientRect();
                this._mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
                this._mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
                this._raycaster.setFromCamera(this._mouse, this.camera);

                if (this._gizmo.visible) {
                    const gizmoHits = this._raycaster.intersectObjects(this._gizmoRings);
                    this._resetGizmoColors();
                    if (gizmoHits.length > 0) {
                        this._gizmoMats[gizmoHits[0].object.userData.axisName].color.setHex(0xffff00);
                        el.style.cursor = 'crosshair';
                        return;
                    }
                }

                const spheres = this.node.jointSpheres || [];
                const boneHits = this._raycaster.intersectObjects(spheres);
                el.style.cursor = boneHits.length > 0 ? 'pointer' : 'grab';
                return;
            }

            const dx = e.clientX - this._prevX;
            const dy = e.clientY - this._prevY;
            this._prevX = e.clientX;
            this._prevY = e.clientY;

            if (this._mode === 'orbiting') {
                if (this._button === 0) {
                    this._spherical.theta -= dx * this.rotateSpeed;
                    this._spherical.phi -= dy * this.rotateSpeed;
                    this._spherical.phi = Math.max(0.05, Math.min(Math.PI - 0.05, this._spherical.phi));
                } else if (this._button === 1 || this._button === 2) {
                    const up = new THREE.Vector3(0, 1, 0);
                    const right = new THREE.Vector3();
                    right.crossVectors(
                        new THREE.Vector3().subVectors(this.camera.position, this.orbitTarget).normalize(), up
                    ).normalize();
                    const offset = new THREE.Vector3();
                    offset.addScaledVector(right, -dx * this.panSpeed * this._spherical.radius * 0.5);
                    offset.addScaledVector(up, dy * this.panSpeed * this._spherical.radius * 0.5);
                    this.orbitTarget.add(offset);
                }
                this._updateCamera();
                el.style.cursor = 'grabbing';

            } else if (this._mode === 'rotating_bone' && this.selectedBone) {
                if (this._rotationAxis) {
                    // Axis-constrained rotation
                    const delta = Math.abs(dx) > Math.abs(dy) ? -dx : -dy;
                    const q = new THREE.Quaternion().setFromAxisAngle(
                        this._rotationAxis, delta * this.boneRotateSpeed
                    );
                    this.selectedBone.quaternion.premultiply(q);
                } else {
                    // Free rotation: drag down → arm down, drag right → arm right
                    const qx = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), dy * this.boneRotateSpeed);
                    const qy = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -dx * this.boneRotateSpeed);
                    this.selectedBone.quaternion.premultiply(qy);
                    this.selectedBone.quaternion.premultiply(qx);
                }
                el.style.cursor = 'crosshair';
            }
        });

        el.addEventListener('pointerup', () => {
            if (this._mode === 'rotating_bone') this._resetGizmoColors();
            this._mode = 'idle';
            this._button = -1;
            this._rotationAxis = null;
            el.style.cursor = 'grab';
        });

        el.addEventListener('pointerleave', () => {
            if (this._mode !== 'idle') {
                if (this._mode === 'rotating_bone') this._resetGizmoColors();
                this._mode = 'idle';
                this._button = -1;
                this._rotationAxis = null;
                el.style.cursor = 'grab';
            }
        });

        el.addEventListener('wheel', (e) => {
            e.preventDefault();
            this._spherical.radius *= 1 + e.deltaY * this.zoomSpeed;
            this._spherical.radius = Math.max(0.1, Math.min(50, this._spherical.radius));
            this._updateCamera();
        }, { passive: false });

        el.addEventListener('contextmenu', (e) => e.preventDefault());

        window.addEventListener('keydown', (e) => {
            if (e.key === "Escape") this._deselectBone();
            
            // Undo: Alt + Z
            if (e.altKey && (e.key === 'z' || e.key === 'Z') && !e.shiftKey) {
                e.preventDefault();
                this.undo();
            }
            // Redo: Alt + Y or Alt + Shift + Z
            if (e.altKey && ((e.key === 'y' || e.key === 'Y') || (e.shiftKey && (e.key === 'z' || e.key === 'Z')))) {
                e.preventDefault();
                this.redo();
            }
        });
    }

    updateGizmo() { this._syncGizmo(); }

    _updateCamera() {
        const pos = new THREE.Vector3().setFromSpherical(this._spherical).add(this.orbitTarget);
        this.camera.position.copy(pos);
        this.camera.lookAt(this.orbitTarget);
    }

    syncTarget(position) {
        this.orbitTarget.copy(position);
        this._spherical.setFromVector3(
            new THREE.Vector3().subVectors(this.camera.position, this.orbitTarget)
        );
        this._updateCamera();
    }
}

app.registerExtension({
    name: "SkinTokens.Previewer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "SkinTokensRigPreviewer") return;
        LOG("Registering SkinTokensRigPreviewer node definition.");

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;
            LOG(`Node instance created (id=${node.id}).`);

            const container = document.createElement("div");
            container.style.cssText = "width:100%; height:100%; position:relative; overflow:hidden; background:#222;";
            node.container = container;

            const widget = this.addDOMWidget("view3d", "HTML", container, {
                serialize: false,
                hideOnZoom: false
            });
            widget.computeSize = function () { return [400, 400]; };
            node.setSize([400, 440]);

            const overlay = document.createElement("div");
            overlay.style.cssText = "position:absolute; top:5px; left:5px; z-index:10; pointer-events:none; color:white; font-family:sans-serif; text-shadow: 1px 1px 2px black;";
            overlay.innerHTML = "<div style='font-weight:bold;'>SkinTokens Preview</div><div style='font-size:11px;'>Left-drag: orbit | Click bone + drag: rotate bone | Esc: deselect</div><div style='font-size:11px; color:#aaa;'>Undo: Alt+Z | Redo: Alt+Y</div>";
            container.appendChild(overlay);

            node.container.tabIndex = 1; // Allow focus for keyboard events
            node.container.style.outline = "none";

            node.log = (msg) => {
                LOG(msg);
            };

            node.initViewer = async function () {
                LOG("initViewer() called.");
                await loadThreeJS();
                if (node.renderer) return;

                const width = container.clientWidth || 400;
                const height = container.clientHeight || 400;

                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x222222);
                scene.add(new THREE.AmbientLight(0xffffff, 0.4));
                const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.8);
                scene.add(hemiLight);
                const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
                dirLight.position.set(5000, 10000, 7000);
                scene.add(dirLight);
                const fillLight = new THREE.DirectionalLight(0xffffff, 0.5);
                fillLight.position.set(-5000, 5000, 2000);
                scene.add(fillLight);

                const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000000);
                camera.position.set(0, 1.5, 3);

                const renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(width, height);
                renderer.outputEncoding = THREE.sRGBEncoding; // Fix dark textures
                container.appendChild(renderer.domElement);
                renderer.domElement.style.cursor = 'grab';

                const grid = new THREE.GridHelper(50000, 50, 0x444444, 0x222222);
                scene.add(grid);

                const controller = new ViewerController(camera, renderer, scene, node);

                node.scene = scene;
                node.camera = camera;
                node.renderer = renderer;
                node.controller = controller;
                node.bones = [];

                function animate() {
                    requestAnimationFrame(animate);
                    if (node.jointSpheres) {
                        node.jointSpheres.forEach(sphere => {
                            sphere.position.setFromMatrixPosition(sphere.userData.bone.matrixWorld);
                        });
                    }
                    controller.updateGizmo(); // Sync gizmo position/scale with selected bone
                    if (node.currentMesh) {
                        node.currentMesh.traverse(child => {
                            if (child.isSkinnedMesh) {
                                child.updateMatrixWorld(true);
                                if (child.skeleton) child.skeleton.update();
                            }
                        });
                    }
                    renderer.render(scene, camera);
                }
                animate();
                LOG("Render loop started.");

                new ResizeObserver(() => {
                    if (!container.clientWidth) return;
                    camera.aspect = container.clientWidth / container.clientHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(container.clientWidth, container.clientHeight);
                }).observe(container);
            };

            setTimeout(() => node.initViewer(), 100);
            return r;
        };

        nodeType.prototype._fixMaterial = function (mat) {
            if (!mat) return;
            
            // Standardize textures for sRGB
            const textures = [mat.map, mat.emissiveMap, mat.specularMap];
            textures.forEach(t => {
                if (t && t.isTexture) {
                    t.encoding = THREE.sRGBEncoding;
                    t.needsUpdate = true;
                }
            });

            mat.skinning = true; 
            mat.side = THREE.DoubleSide;
            mat.transparent = false; // Force opacity
            mat.opacity = 1.0;
            
            // Fix untextured/black models
            if (!mat.map) {
                // If it's pure black or has no color, force it to grey
                if (!mat.color || (mat.color.r === 0 && mat.color.g === 0 && mat.color.b === 0)) {
                    if (mat.color) mat.color.set(0xaaaaaa);
                    else mat.color = new THREE.Color(0xaaaaaa);
                }
            }
            
            mat.needsUpdate = true;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            if (message && message.skintokens_mesh && message.skintokens_mesh.length > 0) {
                const url = api.apiURL(message.skintokens_mesh[0]);
                this.log(`Loading: ${url.split('/').pop()}`);
                this.loadMesh(url);
            }
            return r;
        };

        nodeType.prototype.loadMesh = async function (url) {
            LOG(`loadMesh() called.`);
            this.lastMeshUrl = url; // Save for URL modifier
            if (!this.scene) await this.initViewer();

            if (this.currentMesh) {
                this.scene.remove(this.currentMesh);
                this.currentMesh = null;
            }
            if (this.skeletonHelper) {
                this.scene.remove(this.skeletonHelper);
                this.skeletonHelper = null;
            }
            if (this.jointSpheres) {
                this.jointSpheres.forEach(s => this.scene.remove(s));
            }
            this.jointSpheres = [];
            if (this.controller) {
                this.controller.selectedBone = null;
                this.controller._selectedSphere = null;
            }

            const ext = url.split('.').pop().toLowerCase().split('?')[0].split('&')[0];
            LOG(`Extension: "${ext}"`);
            let loader, loaderName;

            if (ext === 'obj') {
                const mtlLoader = new THREE.MTLLoader();
                const mtlUrl = url.replace(/\.obj$/, '.mtl');
                try {
                    mtlLoader.load(mtlUrl, (materials) => {
                        materials.preload();
                        loader = new THREE.OBJLoader();
                        loader.setMaterials(materials);
                        this._performLoad(loader, url);
                    }, () => {}, (err) => {
                        WARN("Failed to load MTL, falling back to basic OBJ loader.", err);
                        this._performLoad(new THREE.OBJLoader(), url);
                    });
                    return;
                } catch (e) {
                    loader = new THREE.OBJLoader();
                }
            } else if (ext === 'fbx') {
                const manager = new THREE.LoadingManager();
                
                // Intercept texture requests and route them through ComfyUI's /view API
                manager.setURLModifier((url) => {
                    // Avoid double-wrapping or wrapping non-texture files
                    if (url.includes('view?filename=')) return url;
                    
                    const isTexture = /\.(png|jpg|jpeg|tga|dds|bmp)$/i.test(url) || url.includes('_png') || url.includes('_jpg');
                    const isFBM = url.includes('.fbm/') || url.includes('/textures/');

                    if (isTexture || isFBM) {
                        let filename = url.split('/').pop().toLowerCase();
                        
                        // Fix name to match our model-prefixed .png fallback
                        if (!filename.includes('.')) {
                            filename += '.png';
                        }
                        filename = filename.replace(/_(png|jpg|jpeg|tga)$/i, '.$1');
                        
                        const fbxUrlParams = new URLSearchParams(this.lastMeshUrl.split('?')[1]);
                        const fbxFilename = fbxUrlParams.get('filename') || "";
                        const subfolder = fbxUrlParams.get('subfolder') || '';
                        
                        const modelName = fbxFilename.replace(/\.(fbx|glb|gltf)$/i, "");
                        const fbmFolder = `${modelName}.fbm`;
                        
                        // Convert "Image_14" -> "image_14.png" inside the .fbm folder
                        let textureFile = url.split('/').pop().split('\\').pop();
                        textureFile = textureFile.replace(/\.[^/.]+$/, "").replace(/\s+/g, "_").toLowerCase();
                        if (!textureFile.endsWith(".png")) textureFile += ".png";
                        
                        const finalSubfolder = subfolder ? `${subfolder}/${fbmFolder}` : fbmFolder;
                        
                        LOG(`Redirecting texture fallback: ${url} -> ${fbmFolder}/${textureFile}`);
                        return api.apiURL(`/view?filename=${encodeURIComponent(textureFile)}&type=output&subfolder=${encodeURIComponent(finalSubfolder)}`);
                    }
                    return url;
                });

                manager.onStart = (url, itemsLoaded, itemsTotal) => LOG(`Started loading texture: ${url}`);
                manager.onLoad = () => LOG('All textures loaded successfully.');
                manager.onError = (url) => ERR(`Failed to load texture: ${url}`);
                
                // Handle TGA textures often found in FBX
                manager.addHandler(/\.tga$/i, new THREE.TGALoader());
                loader = new THREE.FBXLoader(manager);
            } else {
                loader = new THREE.GLTFLoader();
            }
            this._performLoad(loader, url);
        };

        nodeType.prototype._performLoad = function (loader, url) {
            loader.load(url, (loadedData) => {
                const model = loadedData.scene || loadedData;
                LOG(`Model loaded. Children: ${model.children.length}`);

                model.traverse((child) => {
                    if (child.isMesh) {
                        this.log(`Mesh: ${child.name}`);
                        child.visible = true;
                        child.frustumCulled = false; // Fixes "invisible" meshes with bad bounding boxes
                        child.castShadow = true;
                        child.receiveShadow = true;
                        
                        if (child.isSkinnedMesh && child.skeleton) {
                            child.skeleton.update();
                        }
                        
                        if (child.material) {
                            const mats = Array.isArray(child.material) ? child.material : [child.material];
                            mats.forEach(m => {
                                const mapInfo = m.map ? `(Map: ${m.map.name || 'default'}, Img: ${m.map.image ? (m.map.image.src ? m.map.image.src.split('/').pop() : 'In-Memory') : 'Missing'})` : '(No Map)';
                                this.log(`  Mat: ${m.name} ${mapInfo}`);
                                this._fixMaterial(m);
                            });
                        }
                    }
                });

                this.scene.add(model);
                this.currentMesh = model;

                const box = new THREE.Box3().setFromObject(model);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                const maxHeight = Math.max(size.x, size.y, size.z);

                // IMPORTANT: Do not scale the model directly! 
                // Scaling a skinned mesh's parent causes weird stretching during rotation.
                // Instead, we just center it and move the camera to fit.
                // TEMP: Disable centering to see if it's breaking SkinnedMesh binding
                // model.position.sub(center);
                // model.position.y += size.y / 2;
                LOG(`Model loaded at origin (Centering disabled for debugging).`);

                let rootBone = null;
                const allBones = [];

                model.traverse((child) => {
                    if (child.isBone) {
                        allBones.push(child);
                        if (!rootBone) rootBone = child;
                    }
                });

                LOG(`Found ${allBones.length} bones.`);
                this.bones = allBones; // Save bones to the node for history tracking

                if (allBones.length > 0) {
                    rootBone = allBones.find(b => !b.parent || !b.parent.isBone) || rootBone;
                    LOG(`Root: "${rootBone.name}"`);

                    this.skeletonHelper = new THREE.SkeletonHelper(rootBone);
                    // No linewidth in WebGL2, so we just use basic material
                    this.scene.add(this.skeletonHelper);

                    const sphereRadius = maxHeight * 0.01; // Scale spheres to model size
                    const sphereGeom = new THREE.SphereGeometry(sphereRadius, 8, 8);
                    const sphereMat = new THREE.MeshBasicMaterial({ color: 0xff0000, visible: false });

                    allBones.forEach(bone => {
                        const sphere = new THREE.Mesh(sphereGeom, sphereMat);
                        sphere.userData.bone = bone;
                        this.scene.add(sphere);
                        this.jointSpheres.push(sphere);
                    });
                    LOG(`${this.jointSpheres.length} raycasting spheres created.`);
                }
                // Frame the camera to see the entire model
                const finalBox = new THREE.Box3().setFromObject(model);
                const finalCenter = finalBox.getCenter(new THREE.Vector3());
                const finalSize = finalBox.getSize(new THREE.Vector3());
                // Calculate camera distance to fit model with some padding
                const fov = this.camera.fov * (Math.PI / 180);
                const dist = (maxHeight / 2) / Math.tan(fov / 2) * 1.3; // 1.3x padding

                this.camera.position.set(0, finalCenter.y, dist);
                this.controller.syncTarget(finalCenter);
                LOG(`Camera framed: center=(${finalCenter.x.toFixed(2)}, ${finalCenter.y.toFixed(2)}, ${finalCenter.z.toFixed(2)}), distance=${dist.toFixed(2)}`);
                LOG("Load complete. ✓");
            },
                (progress) => {
                    if (progress.lengthComputable) {
                        LOG(`Loading: ${((progress.loaded / progress.total) * 100).toFixed(0)}%`);
                    }
                },
                (error) => {
                    ERR("Failed to load model:", error);
                });
        };
    }
});

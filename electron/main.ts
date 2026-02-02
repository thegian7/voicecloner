import { app, BrowserWindow, ipcMain, session, systemPreferences } from "electron";
import { ChildProcess, spawn, execSync } from "child_process";
import * as path from "path";
import * as fs from "fs";
import * as http from "http";
import * as net from "net";

let mainWindow: BrowserWindow | null = null;
let splashWindow: BrowserWindow | null = null;
let pythonProcess: ChildProcess | null = null;
let gradioPort = 0;

// --- Paths ---

function isPackaged(): boolean {
  return app.isPackaged;
}

function resourcePath(...parts: string[]): string {
  if (isPackaged()) {
    return path.join(process.resourcesPath, ...parts);
  }
  return path.join(app.getAppPath(), ...parts);
}

function userDataPath(...parts: string[]): string {
  return path.join(app.getPath("userData"), ...parts);
}

function getPythonBin(): string {
  if (process.platform === "win32") {
    return "python.exe";
  }
  return path.join("bin", "python3");
}

function getPythonPath(): string {
  const bin = getPythonBin();
  // On Linux, prefer extracted copy (AppImage mount is ephemeral)
  const extractedPython = userDataPath("python", bin);
  if (fs.existsSync(extractedPython)) {
    return extractedPython;
  }
  // Check for bundled standalone python
  const vendorPython = resourcePath("vendor", "python", bin);
  if (fs.existsSync(vendorPython)) {
    return vendorPython;
  }
  // Fallback to system python
  return process.platform === "win32" ? "python" : "python3";
}

function getVenvPath(): string {
  return userDataPath("venv");
}

function getVenvPython(): string {
  const venv = getVenvPath();
  if (process.platform === "win32") {
    return path.join(venv, "Scripts", "python.exe");
  }
  return path.join(venv, "bin", "python3");
}

// --- Utilities ---

function findFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.listen(0, () => {
      const port = (server.address() as net.AddressInfo).port;
      server.close(() => resolve(port));
    });
    server.on("error", reject);
  });
}

function pollUrl(url: string, intervalMs: number, timeoutMs: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + timeoutMs;
    const check = () => {
      if (Date.now() > deadline) {
        reject(new Error(`Timeout waiting for ${url}`));
        return;
      }
      http
        .get(url, (res) => {
          if (res.statusCode && res.statusCode < 400) {
            resolve();
          } else {
            setTimeout(check, intervalMs);
          }
        })
        .on("error", () => {
          setTimeout(check, intervalMs);
        });
    };
    check();
  });
}

function sendToSplash(channel: string, ...args: unknown[]) {
  if (splashWindow && !splashWindow.isDestroyed()) {
    splashWindow.webContents.send(channel, ...args);
  }
}

// --- GPU Detection ---

function detectGpu(): "nvidia" | "amd" | "mps" | "cpu" {
  if (process.platform === "darwin") {
    return "mps";
  }
  try {
    execSync("nvidia-smi", { stdio: "ignore" });
    return "nvidia";
  } catch {
    // Check for AMD GPU (ROCm)
    try {
      execSync("rocminfo", { stdio: "ignore" });
      return "amd";
    } catch {
      // Also check lspci for AMD GPUs even without rocminfo installed
      try {
        const lspci = execSync("lspci 2>/dev/null", { encoding: "utf-8" });
        if (/AMD.*VGA|Radeon|RDNA/i.test(lspci)) {
          return "amd";
        }
      } catch {
        // lspci not available
      }
    }
    return "cpu";
  }
}

// --- Setup ---

function isSetupComplete(): boolean {
  return fs.existsSync(getVenvPython());
}

async function extractPythonIfNeeded(): Promise<void> {
  // On Linux (AppImage), the mount path is ephemeral, so copy bundled Python
  // to a persistent location in userData
  if (process.platform !== "linux") return;
  const persistentPython = userDataPath("python", "bin", "python3");
  if (fs.existsSync(persistentPython)) return;
  const vendorDir = resourcePath("vendor", "python");
  if (!fs.existsSync(vendorDir)) return;
  const destDir = userDataPath("python");
  sendToSplash("setup:status", "Extracting Python runtime...");
  await runCommand("cp", ["-a", vendorDir, destDir]);
}

async function runSetup(): Promise<void> {
  await extractPythonIfNeeded();
  const basePython = getPythonPath();
  const venvDir = getVenvPath();

  // Step 1: Create venv
  sendToSplash("setup:status", "Creating Python environment...");
  await runCommand(basePython, ["-m", "venv", venvDir]);

  // Step 2: Upgrade pip
  sendToSplash("setup:status", "Upgrading pip...");
  await runCommand(getVenvPython(), ["-m", "pip", "install", "--upgrade", "pip"]);

  // Step 3: Install PyTorch with correct index
  const gpu = detectGpu();
  sendToSplash("setup:status", `Installing PyTorch (${gpu})...`);

  const pipArgs = ["-m", "pip", "install", "-r", resourcePath("python", "requirements.txt")];
  if (gpu === "nvidia") {
    pipArgs.push("--extra-index-url", "https://download.pytorch.org/whl/cu124");
  } else if (gpu === "amd") {
    pipArgs.push("--extra-index-url", "https://download.pytorch.org/whl/rocm6.2");
  } else if (gpu === "cpu" || process.platform !== "darwin") {
    pipArgs.push("--extra-index-url", "https://download.pytorch.org/whl/cpu");
  }
  // macOS (mps) uses default PyPI torch which includes MPS support

  await runCommand(getVenvPython(), pipArgs, (line) => {
    sendToSplash("setup:log", line);
    // Parse pip progress
    const match = line.match(/Downloading .+ \((.+)\)/);
    if (match) {
      sendToSplash("setup:status", `Downloading: ${match[1]}`);
    }
  });

  sendToSplash("setup:status", "Setup complete!");
}

function runCommand(
  cmd: string,
  args: string[],
  onLine?: (line: string) => void
): Promise<void> {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, {
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, VIRTUAL_ENV: getVenvPath() },
    });

    const handleData = (data: Buffer) => {
      const lines = data.toString().split("\n");
      for (const line of lines) {
        if (line.trim() && onLine) {
          onLine(line.trim());
        }
      }
    };

    proc.stdout?.on("data", handleData);
    proc.stderr?.on("data", handleData);

    proc.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`Command failed with code ${code}: ${cmd} ${args.join(" ")}`));
    });

    proc.on("error", reject);
  });
}

// --- Model Download ---

async function downloadModels(): Promise<void> {
  sendToSplash("setup:status", "Downloading models (this may take a while)...");
  const downloadScript = resourcePath("python", "download_models.py");
  await runCommand(getVenvPython(), [downloadScript], (line) => {
    try {
      const event = JSON.parse(line);
      if (event.type === "start") {
        sendToSplash("setup:status", `Downloading ${event.model} model...`);
      } else if (event.type === "done") {
        sendToSplash("setup:status", `${event.model} model ready`);
      } else if (event.type === "error") {
        sendToSplash("setup:status", `Error: ${event.error}`);
      }
    } catch {
      sendToSplash("setup:log", line);
    }
  });
}

// --- Python / Gradio ---

async function startGradio(): Promise<void> {
  gradioPort = await findFreePort();
  const appPy = resourcePath("python", "app.py");

  pythonProcess = spawn(getVenvPython(), [appPy], {
    env: {
      ...process.env,
      GRADIO_SERVER_PORT: String(gradioPort),
      VIRTUAL_ENV: getVenvPath(),
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  pythonProcess.stdout?.on("data", (data: Buffer) => {
    process.stdout.write(`[gradio] ${data}`);
  });
  pythonProcess.stderr?.on("data", (data: Buffer) => {
    process.stderr.write(`[gradio] ${data}`);
  });

  pythonProcess.on("close", (code) => {
    console.log(`Python process exited with code ${code}`);
    pythonProcess = null;
  });

  // Wait for Gradio to be ready
  await pollUrl(`http://127.0.0.1:${gradioPort}`, 500, 120_000);
}

function killPython(): Promise<void> {
  return new Promise((resolve) => {
    if (!pythonProcess) {
      resolve();
      return;
    }

    const proc = pythonProcess;
    const timeout = setTimeout(() => {
      proc.kill("SIGKILL");
    }, 5000);

    proc.on("close", () => {
      clearTimeout(timeout);
      resolve();
    });

    proc.kill("SIGTERM");
  });
}

// --- Windows ---

function createSplashWindow(): BrowserWindow {
  splashWindow = new BrowserWindow({
    width: 500,
    height: 400,
    frame: false,
    resizable: false,
    transparent: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  splashWindow.loadFile(path.join(__dirname, "..", "..", "electron", "splash.html"));
  return splashWindow;
}

function createMainWindow(): BrowserWindow {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    title: "VoiceCloner",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadURL(`http://127.0.0.1:${gradioPort}`);

  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  return mainWindow;
}

// --- IPC ---

ipcMain.handle("setup:download-models", async () => {
  try {
    await downloadModels();
    return { success: true };
  } catch (e) {
    return { success: false, error: String(e) };
  }
});

ipcMain.handle("setup:skip-models", () => {
  return { success: true };
});

// --- App Lifecycle ---

app.whenReady().then(async () => {
  try {
    // Grant microphone permission to Gradio's web content
    if (process.platform === "darwin") {
      await systemPreferences.askForMediaAccess("microphone");
    }
    session.defaultSession.setPermissionRequestHandler((_webContents, permission, callback) => {
      if (permission === "media") {
        callback(true);
        return;
      }
      callback(false);
    });

    if (!isSetupComplete()) {
      createSplashWindow();
      await runSetup();

      // Ask user about model download
      sendToSplash("setup:ask-models", null);

      // Wait for user choice via IPC
      await new Promise<void>((resolve) => {
        ipcMain.once("setup:models-choice", async (_event, choice: string) => {
          if (choice === "download") {
            await downloadModels();
          }
          resolve();
        });
      });
    }

    // Start Gradio
    if (splashWindow && !splashWindow.isDestroyed()) {
      sendToSplash("setup:status", "Starting VoiceCloner...");
    } else {
      createSplashWindow();
      sendToSplash("setup:status", "Starting VoiceCloner...");
    }

    await startGradio();

    // Switch to main window
    createMainWindow();
    if (splashWindow && !splashWindow.isDestroyed()) {
      splashWindow.close();
    }
  } catch (err) {
    console.error("Startup failed:", err);
    if (splashWindow && !splashWindow.isDestroyed()) {
      sendToSplash("setup:error", String(err));
    }
  }
});

app.on("window-all-closed", async () => {
  await killPython();
  app.quit();
});

app.on("before-quit", async () => {
  await killPython();
});

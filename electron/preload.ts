import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("voicecloner", {
  onStatus: (cb: (msg: string) => void) => {
    ipcRenderer.on("setup:status", (_e, msg) => cb(msg));
  },
  onLog: (cb: (msg: string) => void) => {
    ipcRenderer.on("setup:log", (_e, msg) => cb(msg));
  },
  onAskModels: (cb: () => void) => {
    ipcRenderer.on("setup:ask-models", () => cb());
  },
  onError: (cb: (msg: string) => void) => {
    ipcRenderer.on("setup:error", (_e, msg) => cb(msg));
  },
  chooseModels: (choice: string) => {
    ipcRenderer.send("setup:models-choice", choice);
  },
  downloadModels: () => ipcRenderer.invoke("setup:download-models"),
  skipModels: () => ipcRenderer.invoke("setup:skip-models"),
});

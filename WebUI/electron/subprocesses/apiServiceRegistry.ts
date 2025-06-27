import { ApiService, ApiServiceInformation, BackendStatus, LocalSettings } from './service.ts' // Added missing type imports
import { ComfyUiBackendService } from './comfyUIBackendService.ts'
import { AiBackendService } from './aiBackendService.ts'
import { BrowserWindow } from 'electron'
import { appLoggerInstance } from '../logging/logger.ts'
import getPort, { portNumbers } from 'get-port'
import { LlamaCppBackendService } from './llamaCppBackendService.ts'
import { OpenVINOBackendService } from './openVINOBackendService.ts'
import { IpexLLMBackendService } from './ipexLLMBackendService.ts' // 1. Import IpexLLMBackendService

export type backend = 'ai-backend' | 'comfyui-backend' // This type is small, maybe 'ipex-llm-backend' could be added if used elsewhere for strong typing

export interface ApiServiceRegistry {
  register(apiService: ApiService): void
  getRegistered(): ApiService[]
  getRequired(): ApiService[]
  getService(serviceName: string): ApiService | undefined // Added getService declaration
  bootUpAllSetUpServices(): Promise<{ serviceName: string; state: BackendStatus }[]>
  stopAllServices(): Promise<{ serviceName: string; state: BackendStatus }[]>
  getServiceInformation(): ApiServiceInformation[]
}

export class ApiServiceRegistryImpl implements ApiServiceRegistry {
  private registeredServices: ApiService[] = []

  register(apiService: ApiService): void {
    if (this.registeredServices.find(s => s.name === apiService.name)) { // Prevent duplicate registration by name
      appLoggerInstance.warn(`Service with name ${apiService.name} already registered. Skipping.`, 'apiServiceRegistry');
      return;
    }
    this.registeredServices.push(apiService)
  }

  getRegistered(): ApiService[] {
    return this.registeredServices
  }
  getRequired(): ApiService[] {
    // Assuming 'ai-backend' is the primary required service.
    // If IpexLLMBackendService is also strictly required, this logic might need adjustment
    // or the isRequired flag on the service itself handles this.
    const requiredServices = this.registeredServices.filter((item) => item.name === 'ai-backend')
    if (requiredServices.length === 0) { // Changed to allow for cases where it might not be registered yet during initial calls
      appLoggerInstance.warn("Required Service 'ai-backend' not yet registered", 'apiServiceRegistry');
      return []; // Return empty or handle as appropriate
    }
    return requiredServices
  }

  getService(serviceName: string): ApiService | undefined {
    return this.registeredServices.find((item) => item.name === serviceName)
  }

  async bootUpAllSetUpServices(): Promise<{ serviceName: string; state: BackendStatus }[]> {
    const setUpServices = this.registeredServices.filter((item) => item.isSetUp)
    return Promise.all(
      setUpServices.map((service) =>
        service
          .start()
          .then((state) => {
            return { serviceName: service.name, state }
          })
          .catch((e) => {
            const errorMessage = e instanceof Error ? e.message : String(e);
            appLoggerInstance.error(
              `Failed to start service ${service.name} due to ${errorMessage}`,
              'apiServiceRegistry',
              true,
            )
            return { serviceName: service.name, state: 'failed' as BackendStatus }
          }),
      ),
    )
  }

  async stopAllServices(): Promise<{ serviceName: string; state: BackendStatus }[]> {
    appLoggerInstance.info(`stopping all running services`, 'apiServiceRegistry')
    const runningServices = this.registeredServices.filter(
      (item) => item.currentStatus === 'running',
    )
    return Promise.all(
      runningServices.map((service) =>
        service
          .stop()
          .then((state) => {
            appLoggerInstance.info(
              `service ${service.name} now in state ${state}`,
              'apiServiceRegistry',
            )
            return { serviceName: service.name, state }
          })
          .catch((e) => {
            const errorMessage = e instanceof Error ? e.message : String(e);
            appLoggerInstance.error(
              `Failed to stop service ${service.name} due to ${errorMessage}`,
              'apiServiceRegistry',
              true,
            )
            return { serviceName: service.name, state: 'failed' as BackendStatus }
          }),
      ),
    )
  }

  getServiceInformation(): ApiServiceInformation[] {
    return this.getRegistered().map((service) => service.get_info())
  }
}

let instance: ApiServiceRegistryImpl | null = null

export async function aiplaygroundApiServiceRegistry(
  win: BrowserWindow,
  settings: LocalSettings,
): Promise<ApiServiceRegistryImpl> {
  if (!instance) {
    instance = new ApiServiceRegistryImpl()
    // Note: The order of registration might matter if there are dependencies for setup.
    // AiBackendService is likely a foundational one.
    instance.register(
      new AiBackendService(
        'ai-backend', // This is the main backend, potentially a prerequisite for others using its env
        await getPort({ port: portNumbers(59000, 59999) }),
        win,
        settings,
      ),
    )
    instance.register(
      new OpenVINOBackendService(
        'openvino-backend',
        await getPort({ port: portNumbers(29000, 29999) }),
        win,
        settings,
      ),
    )
    instance.register(
      new ComfyUiBackendService(
        'comfyui-backend',
        await getPort({ port: portNumbers(49000, 49999) }),
        win,
        settings,
      ),
    )
    instance.register(
      new LlamaCppBackendService(
        'llamacpp-backend',
        await getPort({ port: portNumbers(39000, 39999) }),
        win,
        settings,
      ),
    )
    // 2. Add IpexLLMBackendService instance
    // Port is hardcoded in IpexLLMBackendService as 59998, so no need for getPort here.
    instance.register(
      new IpexLLMBackendService(
        'ipex-llm-backend', // Name defined in IpexLLMBackendService
        59998, // Port defined in IpexLLMBackendService, pass it to constructor
        win,
        settings,
      ),
    )
  }
  return instance
}

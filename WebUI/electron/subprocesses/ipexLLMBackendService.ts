import { ChildProcess, spawn } from 'node:child_process'
import path from 'node:path'
import * as filesystem from 'fs-extra'
import { existingFileOrError } from './osProcessHelper.ts' // Assuming this is still needed/relevant
import { DeviceService, UvPipService, LongLivedPythonApiService, SetupProgress } from './service.ts' // Added SetupProgress

const serviceFolder = 'service' // Changed: Main service directory
export class IpexLLMBackendService extends LongLivedPythonApiService {
  readonly serviceDir = path.resolve(path.join(this.baseDir, serviceFolder))
  // Option A: Reuse the main ai-backend-env
  readonly pythonEnvDir = path.resolve(path.join(this.baseDir, `ai-backend-env`))
  // using ls_level_zero from default ai-backend env (same as pythonEnvDir in this case)
  readonly deviceService = this.pythonEnvDir // Path to the environment that has oneAPI tools
  readonly name = 'ipex-llm-backend' // Changed
  readonly isRequired = false // Changed

  // Port should match what ipex_llm_web_api.py uses (default 59998)
  override readonly port = 59998
  healthEndpointUrl = `${this.baseUrl}/health`

  readonly lsLevelZero = new DeviceService(this.deviceService) // Initialize with correct path
  readonly uvPip = new UvPipService(this.pythonEnvDir, serviceFolder) // serviceFolder context for uvPip
  readonly python = this.uvPip.python

  serviceIsSetUp(): boolean {
    // Check if the python executable in the shared ai-backend-env exists.
    // Specific check for ipex_llm_web_api.py might be too much if AiBackendService handles its deployment.
    return filesystem.existsSync(this.python.getExePath()) &&
           filesystem.existsSync(path.join(this.serviceDir, 'ipex_llm_web_api.py'))
  }

  isSetUp = this.serviceIsSetUp()

  async *set_up(): AsyncIterable<SetupProgress> {
    this.setStatus('installing')
    this.appLogger.info('setting up service', this.name)

    try {
      yield {
        serviceName: this.name,
        step: 'start',
        status: 'executing',
        debugMessage: 'starting to set up IPEX LLM service environment',
      }
      // Ensure shared environment's Python and oneAPI tools are available
      await this.lsLevelZero.ensureInstalled()
      await this.uvPip.ensureInstalled() // This sets up python in ai-backend-env if not done

      yield {
        serviceName: this.name,
        step: `check dependencies`, // Changed from "install dependencies"
        status: 'executing',
        debugMessage: `Checking/Ensuring dependencies for IPEX LLM Web API`,
      }

      // The main AiBackendService should have installed requirements-ipex-llm.txt.
      // The ipex_llm_web_api.py might also need general python packages from service/requirements.txt
      // (like Flask, apiflask). Assuming AiBackendService also handles service/requirements.txt or
      // that they are compatible and installed into ai-backend-env.
      // If there were specific requirements for *only* ipex_llm_web_api.py not covered by AiBackendService,
      // they would be installed here. For now, we assume existing setup is sufficient.
      // Example: if ipex_llm_web_api.py had its own requirements-ipex-api.txt:
      // const apiRequirements = existingFileOrError(path.join(this.serviceDir, 'requirements-ipex-api.txt'));
      // await this.uvPip.run(['install', '-r', apiRequirements]);

      // For now, we assume that the main requirements.txt in the 'service' folder,
      // which AiBackendService likely processes, covers Flask, etc. for ipex_llm_web_api.py.
      const mainServiceRequirements = existingFileOrError(path.join(this.serviceDir, 'requirements.txt'))
      // This pip install might be redundant if AiBackendService already installed this exact file
      // into the same ai-backend-env. However, running it ensures dependencies are met if AiBackendService didn't.
      await this.uvPip.run(['install', '-r', mainServiceRequirements])


      yield {
        serviceName: this.name,
        step: `dependencies checked/ensured`,
        status: 'executing', // or 'success' if this is the last setup substep
        debugMessage: `Dependencies for IPEX LLM Web API should be met.`,
      }

      this.setStatus('notYetStarted')
      yield {
        serviceName: this.name,
        step: 'end',
        status: 'success',
        debugMessage: `IPEX LLM service environment setup complete.`,
      }
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : String(e);
      this.appLogger.warn(`Set up of service ${this.name} failed due to ${errorMessage}`, this.name, true)
      this.appLogger.warn(`Aborting set up of ${this.name} service environment`, this.name, true)
      this.setStatus('installationFailed')
      yield {
        serviceName: this.name,
        step: 'end',
        status: 'failed',
        debugMessage: `Failed to setup IPEX LLM service environment due to ${errorMessage}`,
      }
    }
  }

  async spawnAPIProcess(): Promise<{
    process: ChildProcess
    didProcessExitEarlyTracker: Promise<boolean>
  }> {
    const additionalEnvVariables = {
      PYTHONNOUSERSITE: 'true',
      SYCL_ENABLE_DEFAULT_CONTEXTS: '1', // Relevant for Intel GPUs
      SYCL_CACHE_PERSISTENT: '1',       // Relevant for Intel GPUs
      PYTHONIOENCODING: 'utf-8',
      ...(await this.lsLevelZero.getDeviceSelectorEnv()), // For device selection
    }

    const apiProcess = spawn(
      this.python.getExePath(),
      ['ipex_llm_web_api.py', '--port', this.port.toString()], // Changed script name
      {
        cwd: this.serviceDir, // Should be 'service' directory
        windowsHide: true,
        env: Object.assign({}, process.env, additionalEnvVariables), // Ensure clean env merge
      },
    )

    const didProcessExitEarlyTracker = new Promise<boolean>((resolve, _reject) => {
      apiProcess.on('error', (error) => {
        this.appLogger.error(`encountered error of process in ${this.name} : ${error.message}`, this.name)
        resolve(true)
      })
      apiProcess.on('exit', (code, signal) => {
        // Non-zero exit code or signal termination is usually an issue for a long-lived service
        if (code !== 0 || signal) {
            this.appLogger.error(`encountered unexpected exit (code: ${code}, signal: ${signal}) in ${this.name}.`, this.name)
            resolve(true)
        } else {
            this.appLogger.info(`${this.name} process exited normally (code: 0). This is unexpected for a long-lived service.`, this.name)
            resolve(true) // Still resolve true as it exited early for a service meant to be long-lived
        }
      })
    })

    return {
      process: apiProcess,
      didProcessExitEarlyTracker: didProcessExitEarlyTracker,
    }
  }
}

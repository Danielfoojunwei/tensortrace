import { defineStore } from 'pinia'
import { ref } from 'vue'

export const usePeftStore = defineStore('peft', () => {
  const step = ref(1)
  const runId = ref(null)
  const pollingInterval = ref(null)

  const run = ref({
    status: 'idle', // idle, pending, running, completed, failed
    progress: 0,
    stage: '',
    logs: [],
    metrics: { loss: 0, accuracy: 0 }
  })

  const config = ref({
    backend: 'local-gpu',
    model: 'llama-3-8b',
    dataset: 'finance-alpaca',
    hparams: {
      lora_r: 16,
      lora_alpha: 32,
      dropout: 0.05
    },
    integrations: {
      wandb: false,
      mlflow: true
    }
  })

  // Start a training run via backend API
  const startRun = async () => {
    run.value.status = 'pending'
    run.value.progress = 0
    run.value.logs = []
    run.value.logs.push('[INIT] Submitting training job to backend...')

    try {
      // Build wizard state from config
      const wizardState = {
        profile_id: 'local-hf',
        model_id: config.value.model,
        dataset_path: config.value.dataset,
        lora_config: {
          r: config.value.hparams.lora_r,
          alpha: config.value.hparams.lora_alpha,
          dropout: config.value.hparams.dropout
        },
        integrations: config.value.integrations
      }

      const res = await fetch('/api/v1/peft/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(wizardState)
      })

      if (res.ok) {
        const data = await res.json()
        runId.value = data.run_id
        run.value.status = 'running'
        run.value.logs.push(`[OK] Run started: ${data.run_id}`)

        // Start polling for status
        startPolling()
      } else {
        throw new Error('Failed to start run')
      }
    } catch (e) {
      console.warn("Backend not available", e)
      run.value.status = 'failed'
      run.value.logs.push('[ERROR] Backend unavailable. Training run was not started.')
    }
  }

  // Poll backend for run status
  const startPolling = () => {
    if (pollingInterval.value) clearInterval(pollingInterval.value)

    pollingInterval.value = setInterval(async () => {
      try {
        const res = await fetch(`/api/v1/peft/runs/${runId.value}`)
        if (res.ok) {
          const data = await res.json()
          run.value.status = data.status.toLowerCase()
          run.value.progress = data.progress || 0
          run.value.stage = data.stage || ''

          if (data.metrics_json) {
            run.value.metrics = data.metrics_json
          }

          // Add stage update to logs
          if (data.stage && !run.value.logs.includes(`[STAGE] ${data.stage}`)) {
            run.value.logs.push(`[STAGE] ${data.stage}`)
          }

          // Stop polling when complete
          if (data.status === 'COMPLETED' || data.status === 'FAILED') {
            clearInterval(pollingInterval.value)
            pollingInterval.value = null
            run.value.logs.push(`[DONE] Training ${data.status.toLowerCase()}`)
          }
        }
      } catch (e) {
        console.warn("Polling failed", e)
      }
    }, 2000)
  }

  // Load profiles from backend
  const loadProfiles = async () => {
    try {
      const res = await fetch('/api/v1/peft/profiles')
      if (res.ok) {
        return await res.json()
      }
    } catch (e) {
      console.warn("Failed to load profiles", e)
    }
    return []
  }

  const applyProfile = async (profileName) => {
    if (profileName === 'local-hf') {
      config.value.backend = 'local-gpu'
      config.value.hparams.lora_r = 64
      run.value.logs.push('[CONFIG] Applied Local HF Profile')
    }

    // Try to fetch profile from backend
    try {
      const profiles = await loadProfiles()
      const profile = profiles.find(p => p.id === profileName)
      if (profile) {
        run.value.logs.push(`[CONFIG] Loaded profile: ${profile.name}`)
      }
    } catch (e) {
      console.warn("Profile load failed", e)
    }
  }

  // List previous runs
  const listRuns = async () => {
    try {
      const res = await fetch('/api/v1/peft/runs')
      if (res.ok) {
        return await res.json()
      }
    } catch (e) {
      console.warn("Failed to list runs", e)
    }
    return []
  }

  // Promote a run to a channel
  const promoteRun = async (runIdToPromote, channel) => {
    try {
      const res = await fetch(`/api/v1/peft/runs/${runIdToPromote}/promote`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ channel })
      })
      return await res.json()
    } catch (e) {
      console.warn("Promote failed", e)
      return { ok: false, reason: 'Network error' }
    }
  }

  return {
    step,
    run,
    runId,
    config,
    startRun,
    applyProfile,
    loadProfiles,
    listRuns,
    promoteRun
  }
})

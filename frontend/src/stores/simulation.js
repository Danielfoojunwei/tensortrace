import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

const STAGE_LABELS = {
  capture: 'Capture',
  embed: 'Embed',
  gate: 'Gate',
  peft: 'PEFT',
  shield: 'Shield',
  sync: 'Sync',
  pull: 'Pull'
}

const STAGE_TYPES = {
  capture: 'trigger',
  embed: 'action',
  gate: 'action',
  peft: 'action',
  shield: 'security',
  sync: 'aggregator',
  pull: 'aggregator'
}

const STAGE_ICONS = {
  capture: 'play',
  embed: 'layers',
  gate: 'activity',
  peft: 'database',
  shield: 'shield',
  sync: 'server',
  pull: 'database'
}

const getDeviceStatus = (lastSeenAt) => {
  if (!lastSeenAt) return 'offline'
  const lastSeen = new Date(lastSeenAt).getTime()
  const minutes = (Date.now() - lastSeen) / 60000
  if (minutes <= 5) return 'online'
  if (minutes <= 30) return 'degraded'
  return 'offline'
}

export const useSimulationStore = defineStore('simulation', () => {
  const activeFleetId = ref(null)
  const roundStatus = ref('idle')
  const currentRound = ref(0)
  const lastMetrics = ref({ latencyMs: null, throughput: null, activeNodes: 0 })
  const devices = ref([])
  const workflow = ref([])
  const loading = ref(false)
  const errorMessage = ref('')

  const fetchTelemetry = async () => {
    loading.value = true
    errorMessage.value = ''
    try {
      const pipelineRes = await fetch('/api/v1/telemetry/pipeline')
      if (!pipelineRes.ok) {
        throw new Error('Pipeline telemetry unavailable')
      }
      const pipelineData = await pipelineRes.json()
      workflow.value = pipelineData.workflow || []
      activeFleetId.value = pipelineData.fleet_id || null
      currentRound.value = pipelineData.summary?.total_events || 0
      roundStatus.value = pipelineData.safe_mode ? 'degraded' : 'active'

      const devicesRes = await fetch('/api/v1/telemetry/devices')
      if (!devicesRes.ok) {
        throw new Error('Device telemetry unavailable')
      }
      const devicesData = await devicesRes.json()
      devices.value = (devicesData.devices || []).map((device) => ({
        id: device.device_id,
        name: device.device_id,
        fleet: device.fleet_id,
        status: getDeviceStatus(device.last_seen_at),
        last_seen_at: device.last_seen_at
      }))

      lastMetrics.value = {
        latencyMs: workflow.value.length
          ? workflow.value.reduce((acc, step) => acc + (step.latency_ms || 0), 0) / workflow.value.length
          : null,
        throughput: pipelineData.summary?.total_events || 0,
        activeNodes: devices.value.length
      }
    } catch (e) {
      console.warn('Telemetry fetch failed', e)
      workflow.value = []
      devices.value = []
      roundStatus.value = 'idle'
      lastMetrics.value = { latencyMs: null, throughput: null, activeNodes: 0 }
      errorMessage.value = 'Telemetry unavailable. Verify API connectivity.'
    }
    loading.value = false
  }

  const pipelineSteps = computed(() =>
    workflow.value.map((step) => ({
      id: `p-${step.stage}`,
      label: STAGE_LABELS[step.stage] || step.stage,
      type: STAGE_TYPES[step.stage] || 'action',
      status: step.status === 'ok' ? 'success' : step.status,
      icon: STAGE_ICONS[step.stage] || 'activity',
      details: {
        metric: 'Latency',
        value: step.latency_ms?.toFixed?.(1) || step.latency_ms || 0,
        unit: 'ms'
      }
    }))
  )

  const graphNodes = computed(() => {
    const nodes = []

    nodes.push({
      id: 'infra-gateway',
      type: 'pipeline',
      position: { x: 400, y: 150 },
      data: {
        label: 'Fleet Gateway',
        type: 'aggregator',
        status: roundStatus.value === 'degraded' ? 'error' : 'success',
        icon: 'server',
        subtitle: activeFleetId.value ? `Fleet: ${activeFleetId.value}` : 'Fleet: unknown',
        details: { metric: 'Active Nodes', value: lastMetrics.value.activeNodes, unit: '' }
      }
    })

    nodes.push({
      id: 'infra-global',
      type: 'pipeline',
      position: { x: 1400, y: 300 },
      data: {
        label: 'Global Model Registry',
        type: 'security',
        status: 'success',
        icon: 'database',
        subtitle: 'Telemetry Aggregation',
        details: { metric: 'Events', value: lastMetrics.value.throughput, unit: '' }
      }
    })

    devices.value
      .filter((device) => !activeFleetId.value || device.fleet === activeFleetId.value)
      .forEach((device, idx) => {
        nodes.push({
          id: device.id,
          type: 'pipeline',
          position: { x: 50, y: 100 + idx * 220 },
          data: {
            label: device.name,
            type: 'trigger',
            icon: 'play',
            subtitle: device.status,
            status: device.status === 'online' ? 'success' : device.status === 'degraded' ? 'running' : 'idle',
            details: { metric: 'Last Seen', value: device.last_seen_at ? 'recent' : 'unknown', unit: '' }
          }
        })
      })

    pipelineSteps.value.forEach((step, idx) => {
      nodes.push({
        id: step.id,
        type: 'pipeline',
        position: { x: 400 + idx * 300, y: 450 },
        data: { ...step, subtitle: step.status.toUpperCase(), round: currentRound.value }
      })
    })

    return nodes
  })

  const graphEdges = computed(() => {
    const edges = []

    devices.value.forEach((device) => {
      edges.push({
        id: `e-${device.id}-gateway`,
        source: device.id,
        target: 'infra-gateway',
        animated: device.status === 'online' || roundStatus.value === 'active',
        style: { stroke: '#30363d', strokeWidth: 2 },
        type: 'step'
      })
    })

    if (pipelineSteps.value.length > 0) {
      edges.push({
        id: 'e-gateway-trigger',
        source: 'infra-gateway',
        target: pipelineSteps.value[0].id,
        animated: roundStatus.value === 'active',
        style: { stroke: '#30363d', strokeWidth: 2 },
        type: 'step'
      })

      for (let i = 0; i < pipelineSteps.value.length - 1; i++) {
        const current = pipelineSteps.value[i]
        const next = pipelineSteps.value[i + 1]
        edges.push({
          id: `e-${current.id}-${next.id}`,
          source: current.id,
          target: next.id,
          animated: roundStatus.value === 'active',
          style: { stroke: '#30363d', strokeWidth: 2 },
          type: 'step'
        })
      }
    }

    edges.push({
      id: 'e-last-global',
      source: pipelineSteps.value.length ? pipelineSteps.value[pipelineSteps.value.length - 1].id : 'infra-gateway',
      target: 'infra-global',
      animated: roundStatus.value === 'active',
      style: { stroke: '#30363d', strokeWidth: 2 },
      type: 'step'
    })

    return edges
  })

  return {
    activeFleetId,
    roundStatus,
    currentRound,
    lastMetrics,
    devices,
    workflow,
    loading,
    errorMessage,
    fetchTelemetry,
    graphNodes,
    graphEdges
  }
})

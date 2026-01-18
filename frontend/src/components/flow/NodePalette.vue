<script setup>
import { computed, ref } from 'vue'
import { 
  X, Search, Play, Scissors, 
  Layers, Lock, Database, Activity, 
  ShieldCheck, Zap, Server
} from 'lucide-vue-next'

const emit = defineEmits(['close', 'select'])

const searchQuery = ref('')
const selectedCategory = ref('all')

const categories = [
  { id: 'all', label: 'All Nodes' },
  { id: 'trigger', label: 'Triggers', color: 'text-orange-500' },
  { id: 'action', label: 'Processing', color: 'text-blue-500' },
  { id: 'security', label: 'Security', color: 'text-emerald-500' },
  { id: 'aggregator', label: 'Aggregation', color: 'text-purple-500' }
]

const nodes = [
  {
    type: 'trigger',
    label: 'Collect Demonstration',
    icon: 'play',
    subtitle: 'Robot Action',
    description: 'Trigger pipeline when a robot completes a task demonstration.',
    data: { event: 'demo_complete' }
  },
  {
    type: 'trigger',
    label: 'Round Interval',
    icon: 'activity',
    subtitle: 'Time-based',
    description: 'Trigger periodically (e.g., every 1 hour).',
    data: { interval: '1h' }
  },
  {
    type: 'action',
    label: 'Gradient Clipper',
    icon: 'scissors',
    subtitle: 'Pre-processing',
    description: 'Clip gradients to prevent exploding values (L2 Norm).',
    data: { threshold: 1.0 }
  },
  {
    type: 'action',
    label: 'Sparsify (Rand-K)',
    icon: 'layers',
    subtitle: 'Compression',
    description: 'Keep only top-k gradients to reduce bandwidth.',
    data: { ratio: 0.01 }
  },
  {
    type: 'security',
    label: 'N2HE Encrypt',
    icon: 'lock',
    subtitle: 'Privacy',
    description: 'Encrypt gradients using Next-to-Homomorphic Encryption.',
    data: { scheme: 'ckks' }
  },
  {
    type: 'security',
    label: 'Compress (2:4)',
    icon: 'database',
    subtitle: 'Optimization',
    description: 'Apply semi-structured 2:4 sparsity pattern.',
    data: { pattern: '2:4' }
  },
  {
    type: 'aggregator',
    label: 'Homomorphic Sum',
    icon: 'server',
    subtitle: 'Server-side',
    description: 'Sum encrypted vectors without decryption.',
    data: { quorum: 3 }
  },
  {
    type: 'aggregator',
    label: 'Outlier Detect',
    icon: 'shield',
    subtitle: 'Resilience',
    description: 'Reject updates outside 3-sigma range.',
    data: { sigma: 3 }
  },
  {
    type: 'aggregator',
    label: 'Deploy Update',
    icon: 'zap',
    subtitle: 'Fleet Ops',
    description: 'Push aggregated weights back to the fleet.',
    data: { strategy: 'canary' }
  }
]

const filteredNodes = computed(() => {
  return nodes.filter(node => {
    const matchesSearch = node.label.toLowerCase().includes(searchQuery.value.toLowerCase()) || 
                          node.description.toLowerCase().includes(searchQuery.value.toLowerCase())
    const matchesCategory = selectedCategory.value === 'all' || node.type === selectedCategory.value
    return matchesSearch && matchesCategory
  })
})

const getIcon = (name) => {
  switch(name) {
    case 'play': return Play
    case 'scissors': return Scissors
    case 'layers': return Layers
    case 'lock': return Lock
    case 'database': return Database
    case 'activity': return Activity
    case 'shield': return ShieldCheck
    case 'zap': return Zap
    case 'server': return Server
    default: return Activity
  }
}

const getTypeColor = (type) => {
  switch(type) {
    case 'trigger': return 'bg-orange-100 text-orange-600'
    case 'action': return 'bg-blue-100 text-blue-600'
    case 'security': return 'bg-emerald-100 text-emerald-600'
    case 'aggregator': return 'bg-purple-100 text-purple-600'
    default: return 'bg-gray-100 text-gray-600'
  }
}
</script>

<template>
  <div class="fixed right-0 top-0 bottom-0 w-80 bg-white shadow-2xl z-50 flex flex-col border-l border-gray-200" @click.stop>
    <!-- Header -->
    <div class="p-4 border-b border-gray-100 flex justify-between items-center bg-gray-50/50">
      <h3 class="font-bold text-gray-800">Add Node</h3>
      <button @click="$emit('close')" class="p-1 hover:bg-gray-200 rounded text-gray-500 transition-colors">
        <X class="w-5 h-5" />
      </button>
    </div>

    <!-- Search -->
    <div class="p-4 border-b border-gray-100">
      <div class="relative">
        <Search class="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
        <input 
          v-model="searchQuery" 
          type="text" 
          placeholder="Search triggers and actions..."
          class="w-full pl-9 pr-4 py-2 bg-gray-100 border-transparent focus:bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-100 rounded-lg text-sm transition-all outline-none"
          autoFocus
        >
      </div>
      
      <!-- Category Tabs -->
      <div class="flex gap-2 mt-3 overflow-x-auto pb-1 no-scrollbar">
        <button 
          v-for="cat in categories" 
          :key="cat.id"
          @click="selectedCategory = cat.id"
          class="px-3 py-1 rounded-full text-xs font-medium whitespace-nowrap transition-colors border"
          :class="selectedCategory === cat.id 
            ? 'bg-gray-800 text-white border-gray-800' 
            : 'bg-white text-gray-600 border-gray-200 hover:border-gray-300'"
        >
          {{ cat.label }}
        </button>
      </div>
    </div>

    <!-- List -->
    <div class="flex-1 overflow-y-auto p-2 space-y-1">
      <div 
        v-for="node in filteredNodes" 
        :key="node.label"
        @click="$emit('select', node)"
        class="group flex items-start p-3 hover:bg-blue-50 rounded-lg cursor-pointer transition-colors border border-transparent hover:border-blue-100"
      >
        <div 
          class="w-10 h-10 rounded-lg flex items-center justify-center shrink-0 mr-3"
          :class="getTypeColor(node.type)"
        >
          <component :is="getIcon(node.icon)" class="w-5 h-5" />
        </div>
        
        <div class="flex-1 min-w-0">
          <div class="flex items-center justify-between">
            <span class="font-semibold text-gray-800 text-sm">{{ node.label }}</span>
            <span class="text-[10px] text-gray-400 font-mono uppercase">{{ node.type }}</span>
          </div>
          <div class="text-xs text-gray-500 mt-0.5 line-clamp-2">{{ node.description }}</div>
        </div>
        
        <div class="opacity-0 group-hover:opacity-100 self-center ml-2 text-blue-500 transition-opacity">
          <Plus class="w-4 h-4" />
        </div>
      </div>
      
      <div v-if="filteredNodes.length === 0" class="p-8 text-center text-gray-400 text-sm">
        No nodes found matching "{{ searchQuery }}"
      </div>
    </div>
  </div>
</template>

<style scoped>
.no-scrollbar::-webkit-scrollbar {
  display: none;
}
.no-scrollbar {
  -ms-overflow-style: none;
  scrollbar-width: none;
}
</style>

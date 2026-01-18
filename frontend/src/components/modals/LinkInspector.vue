<script setup>
import { ref } from 'vue'
import { X, Network, Lock, Sliders, Activity, Trash2, Zap } from 'lucide-vue-next'

const props = defineProps(['edge'])
const emit = defineEmits(['close', 'update', 'delete'])

const config = ref({
  compression: 'zstd',
  bandwidthLimit: 100,
  privacyBudget: 3.5,
  encryption: 'AES-256-GCM'
})

const save = () => {
  emit('update', { id: props.edge.id, config: config.value })
  emit('close')
}

const confirmDelete = () => {
    if (confirm(`Delete this connection? This cannot be undone.`)) {
        emit('delete', props.edge.id)
    }
}
</script>

<template>
  <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm" @click.self="$emit('close')">
    <div class="bg-[#111] border border-[#333] rounded-xl shadow-2xl w-[520px] flex flex-col overflow-hidden">
      <!-- Header -->
      <div class="p-4 border-b border-[#333] flex justify-between items-center bg-[#0d0d0d]">
        <div class="flex items-center gap-3">
           <div class="p-2 bg-primary/10 rounded-lg">
             <Network class="w-5 h-5 text-primary" />
           </div>
           <div>
             <h3 class="font-bold text-white">Link Configuration</h3>
             <div class="text-xs text-gray-500 font-mono flex items-center gap-1">
               {{ edge.source }} <Zap class="w-3 h-3 text-primary" /> {{ edge.target }}
             </div>
           </div>
        </div>
        <button @click="$emit('close')" class="text-gray-500 hover:text-white p-1 hover:bg-[#222] rounded transition-colors">
          <X class="w-5 h-5" />
        </button>
      </div>

      <!-- Body -->
      <div class="p-6 space-y-6">
         <!-- Privacy Budget -->
         <div class="space-y-3">
            <div class="flex justify-between">
                <label class="text-sm font-semibold text-gray-300 flex items-center gap-2">
                    <Lock class="w-4 h-4 text-gray-500" /> Privacy Budget (ε)
                </label>
                <span class="font-mono text-primary font-bold text-lg">{{ config.privacyBudget }}</span>
            </div>
            <input type="range" min="0.1" max="10" step="0.1" v-model="config.privacyBudget" 
                   class="w-full accent-primary h-1.5 bg-[#222] rounded-lg appearance-none cursor-pointer">
            <div class="text-xs text-gray-500">Lower ε guarantees stronger privacy but adds more noise to gradients.</div>
         </div>

         <!-- Compression -->
         <div class="space-y-3">
            <label class="text-sm font-semibold text-gray-300 flex items-center gap-2">
                <Sliders class="w-4 h-4 text-gray-500" /> Compression Algorithm
            </label>
            <div class="grid grid-cols-3 gap-2">
                <button v-for="algo in ['none', 'zstd', 'lz4']" :key="algo"
                        @click="config.compression = algo"
                        class="px-4 py-2.5 border rounded-lg text-xs font-mono uppercase transition-all"
                        :class="config.compression === algo ? 'border-primary bg-primary/10 text-primary' : 'border-[#333] text-gray-500 hover:border-gray-500 hover:text-white'">
                    {{ algo }}
                </button>
            </div>
         </div>

         <!-- QoS -->
         <div class="space-y-3">
             <label class="text-sm font-semibold text-gray-300 flex items-center gap-2">
                <Activity class="w-4 h-4 text-gray-500" /> Bandwidth Cap
             </label>
             <div class="flex items-center gap-3">
                 <input type="number" v-model="config.bandwidthLimit" 
                        class="bg-[#050505] border border-[#333] rounded-lg p-2.5 text-sm text-white w-28 font-mono focus:border-primary outline-none">
                 <span class="text-sm text-gray-500">Mbps</span>
             </div>
         </div>
      </div>

      <!-- Footer -->
      <div class="p-4 border-t border-[#333] bg-[#0d0d0d] flex justify-between items-center">
          <button @click="confirmDelete" class="px-4 py-2 rounded-lg text-sm text-red-400 hover:bg-red-900/20 transition-colors flex items-center gap-2">
              <Trash2 class="w-4 h-4" /> Delete Link
          </button>
          <div class="flex gap-3">
              <button @click="$emit('close')" class="px-4 py-2 rounded-lg text-sm text-gray-400 hover:text-white hover:bg-[#222] transition-colors">Cancel</button>
              <button @click="save" class="px-5 py-2 bg-primary text-black font-bold rounded-lg hover:bg-orange-500 transition-colors">Apply Config</button>
          </div>
      </div>
    </div>
  </div>
</template>

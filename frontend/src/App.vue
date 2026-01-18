<script setup>
import { ref } from 'vue'
import Sidebar from './components/Sidebar.vue'
import Header from './components/Header.vue'

// Consolidated Views
import CommandCenter from './components/CommandCenter.vue'
import ModelsWorkbench from './components/ModelsWorkbench.vue'
import OperationsCenter from './components/OperationsCenter.vue'
import SecurityCenter from './components/SecurityCenter.vue'
import GlobalSettings from './components/GlobalSettings.vue'

// Main navigation state
const activeTab = ref('dashboard')

// Sub-tab states for deep linking
const modelTab = ref('registry')
const operationsTab = ref('fleets')
const securityTab = ref('overview')

// Handle navigation events from child components
const handleNavigate = (target) => {
  if (typeof target === 'object') {
    activeTab.value = target.page
    if (target.tab) {
      switch (target.page) {
        case 'models':
          modelTab.value = target.tab
          break
        case 'operations':
          operationsTab.value = target.tab
          break
        case 'security':
          securityTab.value = target.tab
          break
      }
    }
  } else {
    activeTab.value = target
  }
}
</script>

<template>
  <div class="flex h-screen bg-background text-secondary overflow-hidden">
    <!-- Sidebar -->
    <Sidebar :activeTab="activeTab" @update:activeTab="handleNavigate" />

    <!-- Main Content -->
    <div class="flex-1 flex flex-col ml-64 transition-all duration-300">
      <Header />

      <main class="flex-1 overflow-hidden relative">
        <!-- Content Switcher -->
        <transition name="fade" mode="out-in">
          <!-- Command Center (Dashboard) -->
          <div v-if="activeTab === 'dashboard'" class="h-full overflow-y-auto">
            <CommandCenter @navigate="handleNavigate" />
          </div>

          <!-- Models Workbench -->
          <div v-else-if="activeTab === 'models'" class="h-full overflow-hidden">
            <ModelsWorkbench :initialTab="modelTab" @navigate="handleNavigate" />
          </div>

          <!-- Operations Center -->
          <div v-else-if="activeTab === 'operations'" class="h-full overflow-hidden">
            <OperationsCenter :initialTab="operationsTab" @navigate="handleNavigate" />
          </div>

          <!-- Security Center -->
          <div v-else-if="activeTab === 'security'" class="h-full overflow-hidden">
            <SecurityCenter :initialTab="securityTab" @navigate="handleNavigate" />
          </div>

          <!-- Settings -->
          <div v-else-if="activeTab === 'settings'" class="h-full overflow-y-auto p-6">
            <GlobalSettings />
          </div>

          <!-- Fallback -->
          <div v-else class="h-full overflow-y-auto p-6 flex items-center justify-center">
            <div class="text-gray-500 text-center">
              <div class="text-4xl mb-4">404</div>
              <div>View not found: {{ activeTab }}</div>
            </div>
          </div>
        </transition>
      </main>
    </div>
  </div>
</template>

<style>
.bg-app { background-color: #000000; }
.bg-card { background-color: #0d1117; }
.border-border { border-color: #30363d; }

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>

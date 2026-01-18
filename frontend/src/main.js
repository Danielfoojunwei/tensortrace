import { createApp } from 'vue'
import { createPinia } from 'pinia'
import './style.css'
import App from './App.vue'
import { createRouter, createWebHistory } from 'vue-router'

// Basic router setup
const router = createRouter({
    history: createWebHistory(),
    routes: [
        { path: '/', component: () => import('./App.vue') } // Temporary
    ]
})

const pinia = createPinia()
const app = createApp(App)

app.use(pinia)
app.use(router)
app.mount('#app')

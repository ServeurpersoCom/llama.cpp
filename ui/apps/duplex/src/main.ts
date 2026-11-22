import { mount } from 'svelte';
import App from './App.svelte';
import './app.css';

// Llama UI's dark tokens apply under the dark class: follow the system
const dark = matchMedia('(prefers-color-scheme: dark)');
const applyTheme = () => document.documentElement.classList.toggle('dark', dark.matches);
dark.addEventListener('change', applyTheme);
applyTheme();

mount(App, { target: document.getElementById('app')! });

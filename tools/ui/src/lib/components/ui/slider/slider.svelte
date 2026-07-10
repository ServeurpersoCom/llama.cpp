<script lang="ts">
	import { Slider as SliderPrimitive } from 'bits-ui';
	import { cn } from '$lib/components/ui/utils';

	type Props = Omit<Extract<SliderPrimitive.RootProps, { type: 'single' }>, 'type'> & {
		class?: string;
	};

	let {
		ref = $bindable(null),
		value = $bindable(0),
		min = 0,
		max = 100,
		step = 1,
		class: className,
		...restProps
	}: Props = $props();
</script>

<SliderPrimitive.Root
	bind:ref
	type="single"
	bind:value
	{min}
	{max}
	{step}
	class={cn('relative flex w-full touch-none items-center select-none', className)}
	{...restProps}
>
	{#snippet children()}
		<span class="relative h-1.5 w-full grow overflow-hidden rounded-full bg-muted">
			<SliderPrimitive.Range class="absolute h-full bg-primary" />
		</span>
		<SliderPrimitive.Thumb
			index={0}
			class="block h-4 w-4 shrink-0 rounded-full border border-primary bg-background shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50"
		/>
	{/snippet}
</SliderPrimitive.Root>

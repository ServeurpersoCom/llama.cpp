import { useEffect, useState } from 'react';
import { BASE_URL } from '../Config';
import { useAppContext } from '../utils/app.context';

export default function ModelSelector() {
  const { config, saveConfig } = useAppContext();
  const [models, setModels] = useState<string[]>([]);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await fetch(`${BASE_URL}/v1/models`, {
          headers: {
            ...(config.apiKey
              ? { Authorization: `Bearer ${config.apiKey}` }
              : {}),
          },
        });
        const data = await res.json();
        const list = data?.data?.map((m: { id: string }) => m.id) ?? [];
        setModels(list);
        if (list.length && (!config.model || !list.includes(config.model))) {
          saveConfig({ ...config, model: list[0] });
        }
      } catch (err) {
        console.error('Failed to fetch models', err);
      }
    };
    fetchModels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.apiKey]);

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    saveConfig({ ...config, model: e.target.value });
  };

  return (
    <select
      className="select select-bordered select-sm w-auto"
      value={config.model}
      onChange={handleChange}
    >
      {models.map((m) => (
        <option key={m} value={m}>
          {m}
        </option>
      ))}
    </select>
  );
}

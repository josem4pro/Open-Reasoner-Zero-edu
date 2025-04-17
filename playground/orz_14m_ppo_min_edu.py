"""
SCRIPT EDUCATIVO: Open-Reasoner-Zero con pythia-14m
=================================================
Este script demuestra el entrenamiento de Reinforcement Learning (RL)
usando PPO (Proximal Policy Optimization) con un modelo pequeño.

Comando para ejecutar en una GPU:
DEBUG_MODE=True python -m playground.orz_14m_ppo_mini_edu
"""

import asyncio
import copy
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from itertools import islice, zip_longest
from typing import Any, Awaitable, Callable, List, Optional, Tuple

import numpy as np
import ray
import torch
from loguru import logger
from omegaconf.listconfig import ListConfig
from typing_extensions import override

from orz.exps.examples.ppo.ppo_base_exp import BasePPOExp, BasePPOExpConfig
from orz.ppo import RayPPOTrainer
from orz.ppo.tools.math_utils import is_equal, solution2answer
from orz.ppo.utils import check_reflection_pattern
from playground.zero_setting_base import CustomDataset, EvalCustomDataset

# MODIFICACIÓN EDUCATIVA: Forzar modo debug y reducir pasos/episodios
DEBUG_MODE = True  # Siempre en modo debug para fines educativos
MAX_STEPS = 2      # Número máximo de pasos a ejecutar

# Configuración de archivos y ejecutor
file_name = f"edu_{os.path.splitext(os.path.basename(__file__))[0]}"
executor = ThreadPoolExecutor(max_workers=8)  # Reducido para entornos educativos


def repeatness(s: str):
    """
    Calcula el nivel de repetitividad en una cadena de texto.

    Un valor alto indica respuestas repetitivas (potencial señal de problemas).
    Utiliza algoritmos de sufijos para detección eficiente.
    """
    def ranks(l):
        index = {v: i for i, v in enumerate(sorted(set(l)))}
        return [index[v] for v in l]

    def suffixArray(s):
        line = ranks(s)
        n, k, ans, sa = len(s), 1, line, [0] * len(s)
        while k < n - 1:
            line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
            ans, k = line, k << 1
        for i, k in enumerate(ans):
            sa[k] = i
        return ans, sa

    def lcp(arr, suffixArr, inv_suff):
        n, ans, k = len(arr), [0] * len(arr), 0

        for i in range(n):
            if inv_suff[i] == n - 1:
                k = 0
                continue

            j = suffixArr[inv_suff[i] + 1]
            while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                k += 1

            ans[inv_suff[i]] = k
            if k > 0:
                k -= 1

        return ans

    arr = [ord(i) for i in s]
    n = len(arr)
    if n <= 1:
        return 0
    c, sa = suffixArray(arr)
    cnt = sum(lcp(arr, sa, c))

    return cnt * 2 / (n * (n + 1))


@dataclass
class PPOExpConfig(BasePPOExpConfig):
    """Configuración para el experimento PPO educativo"""
    use_compute_reward_fn: bool = True  # Usar función de recompensa personalizada
    use_orm_score: bool = False         # No usar puntuación ORM

    # MODIFICACIONES EDUCATIVAS
    # ========================
    # Configuración simplificada para fines educativos
    total_num_nodes: int = 1  # Siempre usar un solo nodo para simplicidad

    # Configuración de recursos (simplificada)
    ref_num_nodes: int = 1
    ref_num_gpus_per_node: int = 1
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 1
    critic_num_nodes: int = 1
    critic_num_gpus_per_node: int = 1
    colocate_all: bool = True           # Colocar todo en el mismo GPU
    colocate_critic_reward: bool = True # Colocar crítico y recompensa juntos
    colocate_actor_ref: bool = True     # Colocar actor y referencia juntos
    vllm_num_engines: int = 1           # Un solo motor vLLM
    vllm_tensor_parallel_size: int = 1  # Sin paralelismo de tensores
    adam_offload: bool = False          # No descargar Adam a CPU
    zero_stage: int = 3                 # Nivel ZeRO para optimización de memoria

    # Rutas para checkpoints y logs
    pretrain: Optional[str] = "EleutherAI/pythia-14m"  # Modelo base pequeño de HuggingFace
    reward_pretrain: Optional[str] = None              # Sin modelo de recompensa pre-entrenado
    save_interval: int = 2                             # Guardar cada 2 pasos
    ckpt_path: str = f"orz_ckpt/{file_name}"           # Ruta para checkpoints
    save_path: str = f"orz_ckpt/{file_name}"           # Ruta para guardar resultados
    tensorboard_log_dir: str = f"orz_logs/{file_name}" # Ruta para logs de TensorBoard

    # Datasets (mantenemos los originales para referencia)
    prompt_data: ListConfig = ListConfig(
        [
            "data/orz_math_57k_collected.json",  # Dataset principal de matemáticas
        ]
    )
    eval_prompt_data: ListConfig = ListConfig(
        [
            "data/eval_data/math500.json",       # Dataset de evaluación MATH500
            "data/eval_data/aime2024.json",      # Dataset de evaluación AIME2024
            "data/eval_data/gpqa_diamond.json",  # Dataset de evaluación GPQA Diamond
        ]
    )
    prompt_data_probs: ListConfig = ListConfig([1.0])  # Probabilidades para datasets

    # Configuración PPO modificada para fines educativos
    actor_learning_rate: float = 1e-6   # Tasa de aprendizaje para política
    critic_learning_rate: float = 5e-6  # Tasa de aprendizaje para crítico
    num_warmup_steps: int = 2           # Pasos de calentamiento (reducido)
    enable_prefix_caching: bool = True  # Habilitar caché de prefijos
    update_ref_every_epoch: bool = True # Actualizar referencia cada época
    advantage_normalize: bool = True    # Normalizar ventajas

    # Reducción de parámetros para análisis educativo
    num_episodes: int = 2               # Solo 2 episodios (original: 20)
    rollout_batch_size: int = 8         # Tamaño de batch reducido
    n_samples_per_prompt: int = 2       # Solo 2 muestras por prompt
    micro_rollout_batch_size: int = 16  # Tamaño de micro batch

    # Pasos de actualización
    policy_update_steps: int = 1        # Solo 1 paso de actualización de política
    critic_update_steps: int = 1        # Solo 1 paso de actualización de crítico
    micro_train_batch_size: int = 1     # Micro batch de entrenamiento
    micro_forward_batch_size: int = 1   # Micro batch para forward pass
    freezing_actor_steps: int = -1      # Sin congelación de actor

    # Coeficientes KL (clave en ORZ: sin regularización KL)
    init_kl_coef: float = 0             # Sin coeficiente KL inicial
    kl_loss_coef: float = 0.0           # Sin coeficiente de pérdida KL
    use_kl_loss: bool = False           # No usar KL loss (modificado a False para educación)
    use_kl_estimator_k3: bool = False   # No usar estimador K3 (modificado a False para educación)

    # Evaluación habilitada para fines educativos
    enable_eval: bool = True            # Habilitamos evaluación
    eval_interval: int = 1              # Evaluar en cada paso

    # Configuración de generación
    prompt_max_len: int = 1024          # Longitud máxima de prompt
    packing_max_len: int = 2048         # Longitud máxima de empaquetado
    generate_max_len: int = 1024        # Longitud máxima de generación
    max_len: int = 2048                 # Longitud máxima total
    temperature: float = 1.0            # Temperatura de sampling
    top_p: float = 1.0                  # Top-p para sampling
    top_k: int = -1                     # Sin top-k
    stop: ListConfig = ListConfig(["User:", "Human:", "Assistant:", "</answer>"])  # Tokens de parada

    # Desactivamos GRPO para simplicidad educativa
    use_grpo: bool = False              # No usar GRPO

    # Utilización de memoria y modelo crítico
    gpu_memory_utilization: float = 0.5 # Utilización de memoria GPU reducida
    critic_pretrain: Optional[str] = None  # Inicializar crítico desde cero

    # Parámetros GAE clave en ORZ (valores óptimos según paper)
    gamma: float = 1.0                  # Factor de descuento = 1.0
    lambd: float = 1.0                  # Lambda GAE = 1.0


class CustomRewardTrainer(RayPPOTrainer):
    """
    Entrenador personalizado que implementa la función de recompensa y generación.

    Características clave:
    - Recompensa binaria (1.0 para respuestas correctas, 0.0 para incorrectas)
    - Extracción de respuestas finales mediante regex
    - Cálculo de métricas de calidad (repetitividad, patrones de reflexión)
    """
    @override
    async def custom_reward_fn(
        self,
        prompts: List[str],
        outputs: List[Any],
        extras: List[dict],
        reward_model_fn: Callable[[List[str], List[str]], Awaitable[torch.Tensor]],
    ) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        """
        Función de recompensa personalizada que evalúa las respuestas generadas.

        Esta es una parte crucial del algoritmo ORZ: asigna recompensa 1.0 solo si:
        1. La respuesta es correcta matemáticamente
        2. La generación se detuvo correctamente (llegó a un token de parada)
        """
        # Inicialización de métricas y listas
        scores = []           # Puntuaciones finales (0.0 o 1.0)
        responses = []        # Respuestas generadas
        avg_non_stop_count = 0  # Contador de respuestas sin parada
        pass_at_n_dict = defaultdict(list)  # Éxitos agrupados por prompt
        num_tokens: List[int] = []  # Longitudes de tokens

        # Tareas para cálculo paralelo de métricas de calidad
        @ray.remote(num_cpus=1)
        def get_repeat_score(res):
            """Calcula puntuación de repetitividad (métrica de calidad)"""
            return repeatness(res)

        @ray.remote(num_cpus=1)
        def get_reflection_pattern_score(res):
            """Calcula presencia de patrones de reflexión (métrica importante)"""
            reflection_pattern_dict = check_reflection_pattern(res)
            reflection_pattern_num = sum(reflection_pattern_dict.values())
            return reflection_pattern_num

        # Lanzar tareas Ray para cálculo paralelo
        rep_tasks = []
        for output in outputs:
            response = output["response"]
            rep_tasks.extend([get_repeat_score.remote(response), get_reflection_pattern_score.remote(response)])
        rep_task_results = ray.get(rep_tasks)  # Recoger resultados

        # Procesar resultados de tareas paralelas
        repeat_scores = []
        reflection_pattern_scores = []
        for idx in range(len(outputs)):
            repeat_scores.append(rep_task_results[idx * 2])
            reflection_pattern_scores.append(rep_task_results[idx * 2 + 1])

        # Recopilar respuestas y tokenizarlas
        for output in outputs:
            responses.append(output["response"])
        output_tokens = self._tokenize(responses, self.cfg.generate_max_len, padding=False)["input_ids"]

        # Registrar ejemplo generado para análisis
        self.writer.add_text(
            "generated_raws",
            f"prompts: {prompts[0]}\n\noutputs: {outputs[0]['response']}\n\nfinal_answer: {outputs[0]['final_answer']}\n\nis_correct: {outputs[0]['iscorrect']}\n\nstop_reason: {outputs[0]['stop_reason']}\n\nresponse_token: {len(output_tokens[0])}",
            self.global_step,
        )

        # PUNTO CLAVE: Cálculo de recompensas
        # La función de recompensa es binaria: 1.0 si es correcta y se detuvo bien, 0.0 en otro caso
        for idx in range(len(outputs)):
            prompt, output, out_token = prompts[idx], outputs[idx], output_tokens[idx]
            rep_score, reflection_pattern_score = repeat_scores[idx], reflection_pattern_scores[idx]
            iscorrect = output["iscorrect"]    # Corrección matemática
            stop_reason = output["stop_reason"]  # Razón de parada
            response_token = len(out_token)    # Longitud de tokens

            # Guardar métricas adicionales en output
            output["repeat_score"] = rep_score
            output["reflection_pattern_score"] = reflection_pattern_score

            # Asignar recompensa: solo respuestas correctas Y detenidas correctamente reciben 1.0
            if stop_reason == "stop":
                score = 1.0 if iscorrect else 0.0
            else:
                avg_non_stop_count += 1
                score = 0.0
            scores.append(score)

            # Contabilizar para pass@n y longitudes
            pass_at_n_dict[prompt].append(scores[-1])
            num_tokens.append(response_token)

        # Convertir a arrays para estadísticas
        num_tokens_arr = np.array(num_tokens, dtype=np.float32)
        scores_arr = np.array(scores)
        correct_tokens_arr = np.array([]) if np.all(scores_arr == 0) else np.array(num_tokens_arr[scores_arr == 1])
        incorrect_tokens_arr = np.array([]) if np.all(scores_arr == 1) else np.array(num_tokens_arr[scores_arr == 0])

        # Modificación de recompensa para GRPO (si está habilitado)
        if self.cfg.use_grpo:
            self.writer.add_scalar("grpo_raw_reward", np.mean(scores), self.global_step)
            # Normalizar recompensas dentro de cada prompt
            for i, prompt in enumerate(prompts):
                scores[i] -= np.mean(pass_at_n_dict[prompt])
                if std := np.std(pass_at_n_dict[prompt]) > 0:
                    scores[i] /= std

        # Guardar resultados de generación para análisis
        def dump_results(prompts, outputs, scores):
            saved = []
            for prompt, output, score in zip(prompts, outputs, scores):
                saved.append(dict(prompt=prompt, score=score, outputs=output))
            json.dump(
                saved,
                open(os.path.join(self.cfg.save_path, f"iter{self.global_step}_generation_results.json"), "w"),
                ensure_ascii=False,
                indent=2,
            )

        # Guardar resultados en segundo plano
        global executor
        asyncio.get_event_loop().run_in_executor(
            executor, dump_results, copy.deepcopy(prompts), copy.deepcopy(outputs), copy.deepcopy(scores)
        )

        # Registrar métricas para análisis
        log_dict = {
            "avg_non_stop_count": avg_non_stop_count / len(prompts),
            "avg_repeat_score": sum(repeat_scores) / len(prompts),
            "avg_reflection_pattern_score": sum(reflection_pattern_scores) / len(prompts),
            "avg_pass_at_n": sum(1 for v in pass_at_n_dict.values() if np.sum(v) > 0) / len(pass_at_n_dict),
            "avg_num_tokens": np.mean(num_tokens_arr).item(),
            "std_num_tokens": np.std(num_tokens_arr).item(),
            "avg_correct_num_tokens": 0 if len(correct_tokens_arr) == 0 else np.mean(correct_tokens_arr).item(),
            "std_correct_num_tokens": 0 if len(correct_tokens_arr) == 0 else np.std(correct_tokens_arr).item(),
            "avg_incorrect_num_tokens": 0 if len(incorrect_tokens_arr) == 0 else np.mean(incorrect_tokens_arr).item(),
            "std_incorrect_num_tokens": 0 if len(incorrect_tokens_arr) == 0 else np.std(incorrect_tokens_arr).item(),
        }

        # Registrar estadísticas en TensorBoard
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, self.global_step)

        # Mostrar estadísticas en log
        logging_str = ",".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
        logger.info(logging_str)

        # Registrar histogramas de longitudes
        if len(correct_tokens_arr) > 0:
            self.writer.add_histogram("correct_response_length", correct_tokens_arr, self.global_step)
        if len(incorrect_tokens_arr) > 0:
            self.writer.add_histogram("incorrect_response_length", incorrect_tokens_arr, self.global_step)

        # Crear tensores de puntuación para PPO
        # Esto es crucial: solo el último token recibe la recompensa completa
        score_tensors = []
        for score, output_token in zip(scores, output_tokens):
            score_tensor = torch.zeros(len(output_token))
            if len(output_token) > 0:
                score_tensor[-1] = score  # Solo el último token recibe la recompensa
            score_tensors.append(score_tensor)

        # Eliminar respuestas vacías
        res_prompts = []
        res_responses = []
        res_score_tensors = []
        for prompt, response, score_tensor in zip(prompts, responses, score_tensors):
            if len(response) > 0:
                res_prompts.append(prompt)
                res_responses.append(response)
                res_score_tensors.append(score_tensor)

        # Mensaje educativo sobre el proceso
        logger.info(f"🎓 EDUCATIVO: Recompensas calculadas. Respuestas correctas: {sum(scores)} de {len(scores)}")

        return res_prompts, res_responses, res_score_tensors

    @override
    @torch.no_grad()
    async def generate_vllm(
        self,
        gen_func: Callable[[List[str]], Awaitable[List[str | Any]]],
        prompts: List[str],
        extras: List[dict],
        **kwargs,
    ) -> List[str | Any]:
        """
        Genera respuestas usando vLLM y extrae las respuestas finales.

        Esta función es crucial para el proceso de rollout en PPO:
        1. Genera respuestas completas con vLLM
        2. Extrae las respuestas finales (boxed{...})
        3. Evalúa si son correctas comparando con referencia
        """
        from vllm import SamplingParams

        # Configurar parámetros de sampling
        sampling_params = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            max_tokens=self.cfg.generate_max_len,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
            stop=self.cfg.stop,
        )

        # Generar respuestas con vLLM
        responses, stop_reasons = await gen_func(
            prompts=prompts, sampling_params=sampling_params, use_tqdm=False, truncate_prompt=True
        )

        # Función para extraer respuestas finales en paralelo
        @ray.remote(num_cpus=1)
        def extract_final_answers_batch(responses: List[str]) -> List[str]:
            """Extrae respuestas boxed{...} entre etiquetas <answer>...</answer>"""
            pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
            results = []
            for response in responses:
                matches = re.findall(pattern, response)
                results.append(matches[-1] if matches else "")
            return results

        # Procesar por lotes para eficiencia
        BATCH_SIZE = 16
        num_batches = (len(responses) + BATCH_SIZE - 1) // BATCH_SIZE

        # Extraer respuestas finales (patrón \boxed{...})
        extract_tasks = []
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(responses))
            batch = responses[start_idx:end_idx]
            extract_tasks.append(extract_final_answers_batch.remote(batch))
        batched_results = await asyncio.gather(*[asyncio.to_thread(ray.get, task) for task in extract_tasks])
        final_answers = [answer for batch in batched_results for answer in batch]

        # Verificar corrección matemática
        global executor
        equal_tasks = []
        for extra, final_answer in zip(extras, final_answers):
            equal_tasks.append(is_equal(solution2answer(extra["answer"]), solution2answer(final_answer), executor))
        equal_results = await asyncio.gather(*equal_tasks)

        # Construir resultados finales
        results = []
        for extra, response, final_answer, stop_reason, iscorrect in zip(
            extras, responses, final_answers, stop_reasons, equal_results
        ):
            results.append(
                dict(
                    response=response,
                    iscorrect=iscorrect,
                    stop_reason=stop_reason,
                    final_answer=final_answer,
                )
            )

        # Mensaje educativo sobre generación
        logger.info(f"🎓 EDUCATIVO: Generadas {len(results)} respuestas. Correctas: {sum(r['iscorrect'] for r in results)}")

        return results

    @override
    async def eval(self):
        """
        Función de evaluación en datasets de prueba.

        Evalúa el modelo actual en los datasets de evaluación:
        - MATH500
        - AIME2024
        - GPQA Diamond
        """
        logger.info("🎓 EDUCATIVO: Iniciando evaluación en datasets de prueba")
        from vllm import SamplingParams

        # Configurar parámetros para evaluación
        sampling_params = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.generate_max_len,
            stop=self.cfg.stop,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        )

        from torch.utils.data import DataLoader

        # Cargar dataset de evaluación
        dataset = self.eval_dataset
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, drop_last=False)
        prompt_pre_llm = (len(dataset) + self.cfg.vllm_num_engines - 1) // self.cfg.vllm_num_engines

        # Inicializar resultados y métricas
        output_for_save = []
        log_dict = defaultdict(float)

        # Para cada batch del dataloader
        for batch in dataloader:
            prompts = list(batch[0])
            answers = list(batch[1]["answer"])
            file_names = list(batch[1]["file_name"])
            outputs = []

            # Generar respuestas usando motores vLLM
            for i, llm in enumerate(self.vllm_engines):
                outputs.append(
                    llm.generate.remote(
                        prompts=prompts[i * prompt_pre_llm : (i + 1) * prompt_pre_llm], sampling_params=sampling_params
                    )
                )
            outputs = await asyncio.gather(*outputs)
            outputs = sum(outputs, [])

            # Extraer respuestas finales
            final_answers = []
            pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
            for output in outputs:
                matches = re.findall(pattern, output.outputs[0].text)
                if len(matches) > 0:
                    final_answers.append(matches[-1])
                else:
                    final_answers.append("")

            # Verificar corrección y recopilar estadísticas
            for prompt, output, final_answer, answer, file_name in zip(
                prompts, outputs, final_answers, answers, file_names
            ):
                label = solution2answer(answer)
                prefix_response = solution2answer(final_answer)
                iscorrect = await is_equal(label, prefix_response, executor)
                output_for_save.append(
                    dict(
                        prompt=prompt,
                        output=output.outputs[0].text,
                        final_answer=final_answer,
                        answer=answer,
                        iscorrect=iscorrect,
                    )
                )
                log_dict[f"{file_name}/total_response_len_in_char"] += len(output.outputs[0].text)
                log_dict[f"{file_name}/correct"] += iscorrect
                log_dict[f"{file_name}/total"] += 1

        # Calcular estadísticas por dataset
        all_file_names: List[str] = [
            os.path.splitext(os.path.basename(file_path))[0] for file_path in self.cfg.eval_prompt_data
        ]
        for file_name in all_file_names:
            log_dict[f"{file_name}/response_len_in_char"] = (
                log_dict[f"{file_name}/total_response_len_in_char"] / log_dict[f"{file_name}/total"]
            )
            log_dict[f"{file_name}/accuracy"] = log_dict[f"{file_name}/correct"] / log_dict[f"{file_name}/total"]
            log_dict.pop(f"{file_name}/total_response_len_in_char")
            log_dict.pop(f"{file_name}/correct")
            log_dict.pop(f"{file_name}/total")

        # Calcular precisión promedio
        log_dict["eval_accuracy"] = sum([log_dict[f"{file_name}/accuracy"] for file_name in all_file_names]) / len(
            all_file_names
        )

        # Nombrar archivo de salida con estadísticas
        dump_file_name = f"eval_output_iter{self.global_step}"
        for file_name in all_file_names:
            dump_file_name += f"_{file_name}{log_dict[f'{file_name}/accuracy']:.4f}"
        dump_file_name += ".jsonl"

        # Guardar resultados como JSONL
        with open(
            os.path.join(
                self.cfg.save_path,
                dump_file_name,
            ),
            "w",
        ) as f:
            for item in output_for_save:
                f.write(
                    json.dumps(item, ensure_ascii=False) + "\n",
                )

        # Registrar estadísticas
        logging_str = ",".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
        logger.info(f"🎓 EDUCATIVO: Resultados de evaluación: {logging_str}")
        for k, v in log_dict.items():
            self.writer.add_scalar(f"evals/{k}", v, self.global_step)


class PPOExpEdu(BasePPOExp):
    """
    Versión educativa del experimento PPO con explicaciones y limitaciones.

    Características principales:
    - Limita el número de pasos para análisis
    - Añade mensajes educativos
    - Modifica el método run para detener temprano
    """
    @cached_property
    def trainer(self):
        """Crea el entrenador personalizado"""
        vllm_engines = self.create_inference_engine()
        return CustomRewardTrainer(
            cfg=self.cfg,
            strategy=self.strategy,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            vllm_engines=vllm_engines,
            colocate_pg=self.get_colocate_pg,
        )

    @override
    @cached_property
    def train_dataset(self):
        """
        Carga el dataset de entrenamiento.

        Nota educativa: El dataset contiene problemas matemáticos con soluciones.
        """
        # Cargar diálogos desde archivos
        dialogues = []
        for file_path in self.cfg.prompt_data:
            try:
                with open(file_path, "r") as f:
                    dialogues.extend(json.load(f))
                logger.info(f"🎓 EDUCATIVO: Cargado dataset {file_path} con éxito")
            except FileNotFoundError:
                logger.warning(f"🎓 EDUCATIVO: ¡Archivo no encontrado! {file_path}")
                # --- Inicio de la parte que estaba en el segundo bloque ---
                logger.warning(f"🎓 EDUCATIVO: Para fines educativos, puedes crear un archivo de ejemplo pequeño")
                # Crear dataset mínimo para pruebas
                sample_dialogues = [
                    {
                        "prompt": "Encuentra el valor de x en la ecuación 2x + 3 = 7.",
                        "answer": "\\boxed{2}"
                    },
                    {
                        "prompt": "Calcula el área de un círculo con radio 3.",
                        "answer": "\\boxed{9\\pi}"
                    }
                ]
                dialogues.extend(sample_dialogues)

        logger.info(f"🎓 EDUCATIVO: Procesando {len(dialogues)} problemas (limitado a 10 para modo educativo)")
        # Limitar a 10 problemas para fines educativos
        dialogues = dialogues[:min(len(dialogues), 10)]

        prompts_dataset = CustomDataset(
            dialogues,
            self.tokenizer,
            self.cfg.prompt_max_len,
            self.strategy,
            pretrain_mode=False,
            num_processors=1,
        )
        logger.info(f"🎓 EDUCATIVO: Dataset de entrenamiento listo con {len(prompts_dataset)} problemas")
        return prompts_dataset

    @override
    @cached_property
    def eval_dataset(self):
        """
        Carga el dataset de evaluación.

        Nota educativa: Estos datasets son la referencia estándar para medir capacidad de razonamiento.
        """
        dialogues = []
        for file_path in self.cfg.eval_prompt_data:
            try:
                with open(file_path, "r") as f:
                    loaded_data = json.load(f)
                    for loaded_data_item in loaded_data:
                        loaded_data_item["file_name"] = os.path.splitext(os.path.basename(file_path))[0]
                    # Limitar a máximo 5 problemas por benchmark para modo educativo
                    loaded_data = loaded_data[:min(len(loaded_data), 5)]
                    dialogues.extend(loaded_data)
                logger.info(f"🎓 EDUCATIVO: Cargado dataset de evaluación {file_path} (limitado a 5 problemas)")
            except FileNotFoundError:
                logger.warning(f"🎓 EDUCATIVO: ¡Archivo de evaluación no encontrado! {file_path}")
                if "math500" in file_path:
                    # Crear mini-dataset de evaluación para MATH500
                    sample_data = [
                        {
                            "prompt": "Calcula la derivada de f(x) = x^2 + 3x.",
                            "answer": "\\boxed{2x + 3}",
                            "file_name": "math500"
                        }
                    ]
                    dialogues.extend(sample_data)

        logger.info(f"🎓 EDUCATIVO: Procesando {len(dialogues)} problemas de evaluación")
        prompts_dataset = EvalCustomDataset(
            dialogues,
            self.tokenizer,
            self.cfg.prompt_max_len,
            self.strategy,
            pretrain_mode=False,
            num_processors=1,
        )
        logger.info(f"🎓 EDUCATIVO: Dataset de evaluación listo con {len(prompts_dataset)} problemas")
        return prompts_dataset

    async def run(self):
        """
        Versión educativa del método run que se detiene después de MAX_STEPS pasos.

        Esta función implementa el bucle principal de entrenamiento PPO con límites
        para análisis educativo y mensajes explicativos.
        """
        # Iniciar Ray si no está iniciado
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            logger.info("🎓 EDUCATIVO: Iniciado entorno Ray para computación distribuida")

        # Crear y configurar el entrenador
        trainer = self.trainer

        # Mensaje educativo inicial
        logger.info("\n" + "="*80)
        logger.info("🎓 INICIO DE ENTRENAMIENTO EDUCATIVO OPEN-REASONER-ZERO")
        logger.info("   Algoritmo: PPO (Proximal Policy Optimization)")
        logger.info("   Modelo base: EleutherAI/pythia-14m (14M parámetros)")
        logger.info("   Características clave:")
        logger.info("   - Sin regularización KL (crucial para escalabilidad)")
        logger.info("   - GAE λ=1.0, γ=1.0 (óptimo para RL en razonamiento)")
        logger.info("   - Recompensa binaria simple (1.0 correcto, 0.0 incorrecto)")
        logger.info("="*80 + "\n")

        # Bucle principal de entrenamiento
        for episode in range(self.cfg.num_episodes):
            logger.info(f"🎓 EDUCATIVO: Iniciando episodio {episode+1}/{self.cfg.num_episodes}")

            # Entrenar por un paso
            await trainer.train()

            # Comprobar punto de detención para análisis educativo
            if trainer.global_step >= MAX_STEPS:
                logger.info(f"🎓 EDUCATIVO: Alcanzado límite de {MAX_STEPS} pasos. Deteniendo entrenamiento.")
                # Guardar estado para análisis
                trainer.save_pretrained(os.path.join(self.cfg.save_path, f"step{trainer.global_step}"))
                break

            # Mensajes educativos después de cada paso
            logger.info(f"\n🎓 ANÁLISIS DEL PASO {trainer.global_step}:")
            logger.info("  - Fase de recolección: El modelo genera múltiples respuestas para cada problema")
            logger.info("  - Fase de evaluación: Se calcula recompensa según corrección matemática")
            logger.info("  - Fase de actualización: PPO optimiza la política usando las ventajas estimadas")
            logger.info("  - Sin regularización KL: Permite mayor exploración y optimización")

            # Evaluación periódica
            if self.cfg.enable_eval and trainer.global_step % self.cfg.eval_interval == 0:
                await trainer.eval()

        # Mensaje educativo final
        logger.info("\n" + "="*80)
        logger.info("🎓 ENTRENAMIENTO EDUCATIVO COMPLETADO")
        logger.info("   Conceptos demostrados:")
        logger.info("   1. Entrenamiento PPO sin regularización KL")
        logger.info("   2. Función de recompensa binaria simple")
        logger.info("   3. Extracción y verificación de respuestas matemáticas")
        logger.info("   4. Evaluación en benchmarks estándar")
        logger.info("="*80 + "\n")


if __name__ == "__main__":
    # Mensaje educativo inicial
    print("\n" + "="*80)
    print("🎓 SCRIPT EDUCATIVO: Open-Reasoner-Zero con modelo pequeño (pythia-14m)")
    print("   Este script ejecuta solo unos pasos para análisis y aprendizaje.")
    print("   Limitado a MAX_STEPS={} para fines educativos.".format(MAX_STEPS))
    print("="*80 + "\n")

    # Crear experimento con configuración educativa
    exp = PPOExpEdu().set_cfg(PPOExpConfig())
    logger.info("CONFIGURACIÓN EDUCATIVA:")
    logger.info(exp.get_cfg_as_str(exp.cfg))

    # Crear directorios necesarios
    os.makedirs(exp.cfg.save_path, exist_ok=True)
    os.makedirs(exp.cfg.tensorboard_log_dir, exist_ok=True)
    os.makedirs(exp.cfg.ckpt_path, exist_ok=True)

    # Ejecutar versión educativa
    asyncio.run(exp.run())

    # Mensaje final educativo
    print("\n" + "="*80)
    print("🎓 ANÁLISIS COMPLETADO")
    print("   Resultados:")
    print("   - Logs: {}".format(exp.cfg.tensorboard_log_dir))
    print("   - Archivos de salida: {}".format(exp.cfg.save_path))
    print("   - Checkpoints: {}".format(exp.cfg.ckpt_path))
    print("\n   Para analizar en TensorBoard ejecutar:")
    print("   tensorboard --logdir={}".format(exp.cfg.tensorboard_log_dir))
    print("="*80 + "\n")

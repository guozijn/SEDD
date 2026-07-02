const result = document.querySelector("#result");
const trace = document.querySelector("#trace");
const replayButton = document.querySelector("#replay");
const modelSelect = document.querySelector("#model-select");
const modelMeta = document.querySelector("#model-meta");
const promptInput = document.querySelector("#viz-prompt");
const promptExamples = document.querySelector("#prompt-examples");
let activeTrace = null;
let playbackTimer = null;
let availableModels = [];

const EXAMPLE_PROMPTS = [
  {
    label: "Fever",
    prompt:
      "Answer the science multiple-choice question. Return only the final choice as `Answer: <letter>`.\n\n" +
      "Question: Which factor will most likely cause a person to develop a fever?\n" +
      "Choices:\n" +
      "A. a leg muscle relaxing after exercise\n" +
      "B. a bacterial population in the bloodstream\n" +
      "C. several viral particles on the skin\n" +
      "D. carbohydrates being digested in the stomach",
  },
  {
    label: "Circuit",
    prompt:
      "Answer the science multiple-choice question. Return only the final choice as `Answer: <letter>`.\n\n" +
      "Question: A student builds a circuit with a battery, wires, and a bulb. The bulb does not light. Which change would most likely allow the bulb to light?\n" +
      "Choices:\n" +
      "A. opening the switch farther\n" +
      "B. completing the path for electric current\n" +
      "C. replacing the wires with string\n" +
      "D. removing the battery from the circuit",
  },
  {
    label: "Moon",
    prompt:
      "Answer the science multiple-choice question. Return only the final choice as `Answer: <letter>`.\n\n" +
      "Question: Why does the Moon appear to have different shapes during a month?\n" +
      "Choices:\n" +
      "A. Earth blocks different amounts of sunlight from reaching the Moon each night\n" +
      "B. The Moon changes its actual shape as it moves around Earth\n" +
      "C. Different portions of the Moon's sunlit half are visible from Earth\n" +
      "D. Clouds cover different parts of the Moon at regular times",
  },
  {
    label: "Erosion",
    prompt:
      "Answer the science multiple-choice question. Return only the final choice as `Answer: <letter>`.\n\n" +
      "Question: Which process most directly breaks rocks into smaller pieces over time?\n" +
      "Choices:\n" +
      "A. weathering\n" +
      "B. condensation\n" +
      "C. evaporation\n" +
      "D. photosynthesis",
  },
];

modelSelect.disabled = true;
modelSelect.innerHTML = '<option value="">Loading models...</option>';

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

function setBusy(button, busy) {
  button.disabled = busy;
  button.textContent = busy ? "Running..." : button.dataset.label;
}

document.querySelector("#visualize").dataset.label = "Run diffusion trace";

function renderPromptExamples() {
  if (!promptExamples) {
    return;
  }
  promptExamples.innerHTML = "";
  EXAMPLE_PROMPTS.forEach((example) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = example.label;
    button.addEventListener("click", () => {
      promptInput.value = example.prompt;
      promptInput.focus();
    });
    promptExamples.appendChild(button);
  });
}

function clearPlayback() {
  if (playbackTimer) {
    window.clearInterval(playbackTimer);
    playbackTimer = null;
  }
}

function noiseGlyph(sampleIndex, tokenIndex, stepIndex) {
  const glyphs = ["#", "%", "?", "*", "+", "=", "~", ".", "0", "1"];
  return glyphs[(sampleIndex * 17 + tokenIndex * 7 + stepIndex * 5 + Date.now()) % glyphs.length];
}

function tokenClass(value, previousValue) {
  if (value === "[MASK]") {
    return "latent";
  }
  if (previousValue === "[MASK]") {
    return "resolved";
  }
  return "stable";
}

function renderFrame(data, frameIndex) {
  const step = data.steps[frameIndex];
  const previous = data.steps[Math.max(0, frameIndex - 1)];
  const totalFrames = Math.max(1, data.steps.length - 1);
  const progress = Math.round((frameIndex / totalFrames) * 100);
  const label = modelSelect.options[modelSelect.selectedIndex]?.textContent || data.backend;
  result.textContent = `${label} | batch ${data.batch_size} | step ${step.step}/${totalFrames}`;
  trace.innerHTML = "";

  const stage = document.createElement("section");
  stage.className = "denoise-stage";

  const head = document.createElement("div");
  head.className = "stage-head";
  const title = document.createElement("div");
  title.className = "stage-title";
  title.textContent = frameIndex === totalFrames ? "Denoised samples" : "Reverse diffusion in progress";
  const meter = document.createElement("div");
  meter.className = "progress-track";
  const fill = document.createElement("div");
  fill.className = "progress-fill";
  fill.style.width = `${progress}%`;
  meter.appendChild(fill);
  head.appendChild(title);
  head.appendChild(meter);
  stage.appendChild(head);

  const lanes = document.createElement("div");
  lanes.className = "sample-lanes";
  step.samples.forEach((sample, sampleIndex) => {
    const lane = document.createElement("article");
    lane.className = "sample-lane";
    const laneHead = document.createElement("div");
    laneHead.className = "lane-head";
    laneHead.textContent = `sample ${sampleIndex + 1} | latent slots ${sample.masked}`;
    const row = document.createElement("div");
    row.className = "denoise-row";
    sample.tokens.forEach((value, tokenIndex) => {
      const previousValue = previous.samples[sampleIndex]?.tokens[tokenIndex] || "[MASK]";
      const token = document.createElement("span");
      const state = tokenClass(value, previousValue);
      token.className = `denoise-token ${state}`;
      token.textContent = value === "[MASK]" ? noiseGlyph(sampleIndex, tokenIndex, frameIndex) : value || " ";
      row.appendChild(token);
    });
    const text = document.createElement("div");
    text.className = "lane-text";
    text.textContent = sample.text || " ";
    lane.appendChild(laneHead);
    lane.appendChild(row);
    lane.appendChild(text);
    lanes.appendChild(lane);
  });
  stage.appendChild(lanes);
  trace.appendChild(stage);
}

function playTrace(data) {
  clearPlayback();
  activeTrace = data;
  replayButton.disabled = false;
  let frameIndex = 0;
  renderFrame(data, frameIndex);
  playbackTimer = window.setInterval(() => {
    frameIndex += 1;
    if (frameIndex >= data.steps.length) {
      clearPlayback();
      return;
    }
    renderFrame(data, frameIndex);
  }, 720);
}

function renderTrace(data) {
  playTrace(data);
}

function selectedModel() {
  return availableModels.find((model) => model.id === modelSelect.value);
}

function updateModelMeta() {
  const model = selectedModel();
  if (!model) {
    modelMeta.textContent = "";
    return;
  }
  const description = model.description || model.model_path || model.checkpoint || model.backend || "";
  if (availableModels.length <= 1) {
    modelMeta.textContent =
      "Only one model is exposed by the API. Start the app from the remote repo where runs/arc_models/registry.json exists, or pass MODEL_REGISTRY=runs/arc_models/registry.json.";
    return;
  }
  modelMeta.textContent = description;
}

function showModelError(message) {
  availableModels = [];
  modelSelect.innerHTML = "";
  const option = document.createElement("option");
  option.value = "";
  option.textContent = "Models unavailable";
  modelSelect.appendChild(option);
  modelSelect.disabled = true;
  modelMeta.textContent = message;
}

function renderModelOptions(payload) {
  const models = Array.isArray(payload.models) ? payload.models : [];
  modelSelect.innerHTML = "";
  if (models.length === 0) {
    showModelError("The API returned no model entries from /models.");
    return;
  }

  availableModels = models;
  availableModels.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.label || model.id;
    if (model.id === payload.default_model_id) {
      option.selected = true;
    }
    modelSelect.appendChild(option);
  });
  modelSelect.disabled = availableModels.length <= 1;
  updateModelMeta();
}

document.querySelector("#visualize").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  setBusy(button, true);
  try {
    const data = await postJson("/visualize", {
      model_id: modelSelect.value || null,
      prompt: promptInput.value,
      batch_size: Number(document.querySelector("#viz-batch").value),
      max_new_tokens: Number(document.querySelector("#viz-tokens").value),
      steps: Number(document.querySelector("#viz-steps").value),
      temperature: 0.9,
      top_k: 50,
      top_p: 0.95,
    });
    renderTrace(data);
  } catch (error) {
    trace.innerHTML = "";
    result.textContent = String(error);
  } finally {
    setBusy(button, false);
  }
});

document.querySelector("#clear").addEventListener("click", () => {
  clearPlayback();
  activeTrace = null;
  replayButton.disabled = true;
  result.textContent = "";
  trace.innerHTML = "";
});

replayButton.addEventListener("click", () => {
  if (activeTrace) {
    playTrace(activeTrace);
  }
});

fetch("/models")
  .then((response) => {
    if (!response.ok) {
      throw new Error(`/models returned HTTP ${response.status}`);
    }
    return response.json();
  })
  .then(renderModelOptions)
  .catch((error) => {
    showModelError(`Cannot load /models. ${error.message}`);
  });

modelSelect.addEventListener("change", updateModelMeta);
renderPromptExamples();

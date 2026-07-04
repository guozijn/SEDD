const result = document.querySelector("#result");
const trace = document.querySelector("#trace");
const replayButton = document.querySelector("#replay");
const modelSelect = document.querySelector("#model-select");
const modelMeta = document.querySelector("#model-meta");
const promptInput = document.querySelector("#viz-prompt");
const promptLabel = document.querySelector("#prompt-label");
const promptExamples = document.querySelector("#prompt-examples");
const tokenControlLabel = document.querySelector("#token-control-label");
const visualizeButton = document.querySelector("#visualize");
const modeButtons = Array.from(document.querySelectorAll("[data-mode]"));
let activeTrace = null;
let playbackTimer = null;
let availableModels = [];
let activeMode = "infill";

const GENERATION_EXAMPLES = [
  {
    label: "Fever",
    prompt:
      "User: Answer the science multiple-choice question. Return only the final choice as `Answer: <letter>`.\n\n" +
      "Question: Which factor will most likely cause a person to develop a fever?\n" +
      "Choices:\n" +
      "A. a leg muscle relaxing after exercise\n" +
      "B. a bacterial population in the bloodstream\n" +
      "C. several viral particles on the skin\n" +
      "D. carbohydrates being digested in the stomach\n" +
      "Assistant: ",
  },
  {
    label: "Circuit",
    prompt:
      "User: Answer the science multiple-choice question. Return only the final choice as `Answer: <letter>`.\n\n" +
      "Question: A student builds a circuit with a battery, wires, and a bulb. The bulb does not light. Which change would most likely allow the bulb to light?\n" +
      "Choices:\n" +
      "A. opening the switch farther\n" +
      "B. completing the path for electric current\n" +
      "C. replacing the wires with string\n" +
      "D. removing the battery from the circuit\n" +
      "Assistant: ",
  },
  {
    label: "Moon",
    prompt:
      "User: Answer the science multiple-choice question. Return only the final choice as `Answer: <letter>`.\n\n" +
      "Question: Why does the Moon appear to have different shapes during a month?\n" +
      "Choices:\n" +
      "A. Earth blocks different amounts of sunlight from reaching the Moon each night\n" +
      "B. The Moon changes its actual shape as it moves around Earth\n" +
      "C. Different portions of the Moon's sunlit half are visible from Earth\n" +
      "D. Clouds cover different parts of the Moon at regular times\n" +
      "Assistant: ",
  },
];

const INFILL_EXAMPLES = [
  {
    label: "One mask",
    prompt:
      "A bow and arrow is a traditional weapon that [MASK] and can be fired by a trained archer.",
  },
  {
    label: "Two masks",
    prompt:
      "The patient developed a [MASK] after bacteria entered the bloodstream, so the doctor checked for [MASK].",
  },
];

modelSelect.disabled = true;
modelSelect.innerHTML = '<option value="">Loading models...</option>';
visualizeButton.dataset.label = "Run infill trace";

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

function examplesForMode() {
  return activeMode === "infill" ? INFILL_EXAMPLES : GENERATION_EXAMPLES;
}

function renderPromptExamples() {
  promptExamples.innerHTML = "";
  examplesForMode().forEach((example) => {
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

function setMode(mode, { resetPrompt = true } = {}) {
  activeMode = mode;
  modeButtons.forEach((button) => {
    const isActive = button.dataset.mode === mode;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  });
  promptLabel.textContent = mode === "infill" ? "Text with masks" : "Prompt";
  tokenControlLabel.textContent = mode === "infill" ? "Mask tokens" : "Tokens";
  visualizeButton.dataset.label = mode === "infill" ? "Run infill trace" : "Run diffusion trace";
  visualizeButton.textContent = visualizeButton.dataset.label;
  renderPromptExamples();
  if (resetPrompt) {
    promptInput.value = examplesForMode()[0].prompt;
  }
  selectPreferredModel();
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

function visibleTokenText(value, sampleIndex, tokenIndex, frameIndex) {
  if (value === "[MASK]") {
    return noiseGlyph(sampleIndex, tokenIndex, frameIndex);
  }
  if (value === "<eos>") {
    return "";
  }
  return value || "";
}

function appendToken(parent, value, previousValue, sampleIndex, tokenIndex, frameIndex) {
  const text = visibleTokenText(value, sampleIndex, tokenIndex, frameIndex);
  if (!text && value !== "[MASK]") {
    return;
  }
  const token = document.createElement("span");
  token.className = `inline-token ${tokenClass(value, previousValue)}`;
  token.textContent = text || " ";
  parent.appendChild(token);
}

function generatedSegments(sample) {
  return Array.isArray(sample.segments)
    ? sample.segments.filter((segment) => segment.kind === "generated")
    : [];
}

function renderSampleText(sample, previousSample, sampleIndex, frameIndex) {
  const text = document.createElement("div");
  text.className = "inline-text";

  if (Array.isArray(sample.segments)) {
    let generatedIndex = 0;
    const previousGenerated = generatedSegments(previousSample || {});
    sample.segments.forEach((segment) => {
      if (segment.kind === "fixed") {
        const fixed = document.createElement("span");
        fixed.className = "fixed-text";
        fixed.textContent = segment.text || "";
        text.appendChild(fixed);
        return;
      }

      const span = document.createElement("span");
      span.className = "generated-span";
      const previousTokens = previousGenerated[generatedIndex]?.tokens || [];
      const tokens = Array.isArray(segment.tokens) ? segment.tokens : [];
      tokens.forEach((value, tokenIndex) => {
        appendToken(
          span,
          value,
          previousTokens[tokenIndex] || "[MASK]",
          sampleIndex,
          generatedIndex * 100 + tokenIndex,
          frameIndex,
        );
      });
      text.appendChild(span);
      generatedIndex += 1;
    });
    return text;
  }

  const previousTokens = Array.isArray(previousSample?.tokens) ? previousSample.tokens : [];
  const tokens = Array.isArray(sample.tokens) ? sample.tokens : [];
  tokens.forEach((value, tokenIndex) => {
    appendToken(text, value, previousTokens[tokenIndex] || "[MASK]", sampleIndex, tokenIndex, frameIndex);
  });
  return text;
}

function renderFrame(data, frameIndex) {
  const step = data.steps[frameIndex];
  const previous = data.steps[Math.max(0, frameIndex - 1)];
  const totalFrames = Math.max(1, data.steps.length - 1);
  const progress = Math.round((frameIndex / totalFrames) * 100);
  const label = modelSelect.options[modelSelect.selectedIndex]?.textContent || data.backend;
  const mode = data.mode === "infill" ? "infill" : "generation";
  result.textContent = `${label} | ${mode} | batch ${data.batch_size} | step ${step.step}/${totalFrames}`;
  trace.innerHTML = "";

  const stage = document.createElement("section");
  stage.className = "denoise-stage";

  const head = document.createElement("div");
  head.className = "stage-head";
  const title = document.createElement("div");
  title.className = "stage-title";
  title.textContent = frameIndex === totalFrames ? "Denoised text" : "Reverse diffusion in progress";
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
    laneHead.textContent = `sample ${sampleIndex + 1} | unresolved ${sample.masked}`;
    lane.appendChild(laneHead);
    lane.appendChild(renderSampleText(sample, previous.samples[sampleIndex], sampleIndex, frameIndex));
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

function selectedModel() {
  return availableModels.find((model) => model.id === modelSelect.value);
}

function selectPreferredModel() {
  if (availableModels.length === 0) {
    return;
  }
  const preferredId = activeMode === "infill" ? "base" : "arc_lora_sft";
  if (availableModels.some((model) => model.id === preferredId)) {
    modelSelect.value = preferredId;
    updateModelMeta();
  }
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
  selectPreferredModel();
  updateModelMeta();
}

visualizeButton.addEventListener("click", async (event) => {
  const button = event.currentTarget;
  setBusy(button, true);
  try {
    const common = {
      model_id: modelSelect.value || null,
      batch_size: Number(document.querySelector("#viz-batch").value),
      steps: Number(document.querySelector("#viz-steps").value),
      temperature: activeMode === "infill" ? 0.8 : 0.7,
      top_k: activeMode === "infill" ? 40 : 20,
      top_p: activeMode === "infill" ? 0.95 : 0.9,
    };
    const data =
      activeMode === "infill"
        ? await postJson("/visualize-infill", {
            ...common,
            text: promptInput.value,
            tokens_per_mask: Number(document.querySelector("#viz-tokens").value),
          })
        : await postJson("/visualize", {
            ...common,
            prompt: promptInput.value,
            max_new_tokens: Number(document.querySelector("#viz-tokens").value),
          });
    playTrace(data);
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

modeButtons.forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
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
setMode("infill");

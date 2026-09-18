import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/PanCIA";
const skillDir = "/Users/sg2162/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations";
const tmpDir = path.join(workspaceDir, ".codex-slide-build/strategy6-weaknesses");
const finalPath = path.join(workspaceDir, "strategy6-slide-output/strategy6-weaknesses.pptx");
const runtimePython = "/Users/sg2162/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const { resolvePresentationFont, finalizePresentation } = await import(
  pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href
);

await fs.mkdir(tmpDir, { recursive: true });
await fs.mkdir(path.dirname(finalPath), { recursive: true });
const font = resolvePresentationFont();
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const slide = presentation.slides.add();
slide.background.fill = "#F6F3ED";

function box(x, y, w, h, fill = "none") {
  return slide.shapes.add({
    geometry: "rect",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { fill: "none", width: 0 },
  });
}

function text(value, x, y, w, h, size, color, bold = false, align = "left") {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = {
    typeface: font,
    fontSize: size,
    bold,
    color,
    alignment: align,
    verticalAlignment: "middle",
    autoFit: "shrinkText",
  };
  return shape;
}

const navy = "#172A3A";
const coral = "#D55745";
const teal = "#357F82";
const muted = "#66727A";
const rule = "#C9C4BA";

box(0, 0, 1280, 12, coral);
text("Strategy 6: limitations of contextual-bandit survival fusion", 62, 38, 1156, 58, 38, navy, true);
text("The policy and survival experts form a coupled, path-dependent optimization problem", 64, 99, 1100, 30, 18, muted);
box(64, 142, 1152, 2, rule);

text("STATISTICAL LIMITS", 64, 162, 510, 28, 15, teal, true);
text("OPTIMIZATION AND DEPLOYMENT LIMITS", 686, 162, 530, 28, 15, coral, true);
box(640, 162, 2, 418, rule);

const left = [
  ["01", "Objective conflict", "Exploration rewards diversity while exploitation rewards current survival fit. Loss weights and temperature schedules can drive the learned policy."],
  ["02", "Incomplete censored evidence", "Cox partial likelihood uses censored patients only while they remain in risk sets. It provides no individual censored-data likelihood or baseline hazard."],
  ["03", "No performance guarantee", "Personalized routing can underperform the strongest unimodal expert. Better training fit does not guarantee better out-of-sample concordance."],
];
const right = [
  ["04", "Path-dependent EM", "Policy and weighted experts inherit the previous iteration. Non-convex updates depend on initialization and do not yield a unique solution."],
  ["05", "Self-reinforcing feedback", "Early routing errors alter expert weights. The altered experts then generate the risks used to train the next policy."],
  ["06", "Unstable hard decisions", "Near-tied softmax probabilities become discontinuous argmax actions. Small changes can switch experts and materially change pairwise rankings."],
];

function issueColumn(items, x, accent) {
  items.forEach((item, index) => {
    const y = 205 + index * 123;
    text(item[0], x, y, 48, 34, 23, accent, true);
    text(item[1], x + 58, y - 2, 470, 33, 21, navy, true);
    text(item[2], x + 58, y + 34, 468, 69, 16, muted);
    if (index < items.length - 1) box(x + 58, y + 111, 458, 1, rule);
  });
}

issueColumn(left, 64, teal);
issueColumn(right, 686, coral);

box(64, 607, 1152, 72, navy);
text("Consequence", 86, 619, 142, 26, 15, "#E8B96B", true);
text("Strong average fusion can coexist with weak personalization, unstable assignments and high computational cost.", 228, 615, 952, 38, 21, "#FFFFFF", true);

slide.speakerNotes.textFrame.setText(
  "Strategy 6 critique based on the implemented contextual-bandit survival pipeline. " +
  "Technical clarification: Cox partial likelihood does not fully ignore censored patients; " +
  "they contribute while present in risk sets, but it does not provide an individual full censored-data likelihood."
);

const stagingDir = path.join(workspaceDir, ".codex-finalizer-strategy6");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

await finalizePresentation({
  explicitTotalSlideCount: 1,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: runtimePython,
  integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-heading-fit",
  ],
  fontPolicy: { basis: "design", families: [font] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "strategy6-weaknesses.validation.json"),
});

console.log(finalPath);

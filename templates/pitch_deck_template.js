/**
 * ============================================================
 * HACKATHON PITCH DECK TEMPLATE
 * ============================================================
 * 
 * USAGE: Edit the CONFIG section below, then run:
 *   node pitch_deck_template.js
 * 
 * Generates a professional 9-slide pitch deck.
 * Customize the content, keep the structure.
 * ============================================================
 */

const pptxgen = require("pptxgenjs");

// =============================================================
// ✏️  CONFIG — EDIT THIS SECTION
// =============================================================

const CONFIG = {
  teamName: "Team Name",
  projectName: "Project Name",
  tagline: "One-line description of what you built",
  casePartner: "Microsoft / N26",  // Which case you tackled
  
  // Slide 1: Hook / Problem
  hookStat: "72%",
  hookStatLabel: "of [users] struggle with [problem]",
  problemDescription: "Brief description of the pain point your solution addresses",
  
  // Slide 2: Solution
  solutionOneLiner: "We built [PRODUCT] — an AI-powered tool that [DOES WHAT] for [WHO]",
  keyFeatures: [
    { title: "Feature 1", desc: "Brief description" },
    { title: "Feature 2", desc: "Brief description" },
    { title: "Feature 3", desc: "Brief description" },
  ],
  
  // Slide 3: Demo (intentionally minimal — you'll do live demo here)
  demoUrl: "https://your-app.vercel.app",
  
  // Slide 4: How it works
  flowSteps: [
    { step: "1", title: "Input", desc: "User provides X" },
    { step: "2", title: "Process", desc: "AI analyzes with Y" },
    { step: "3", title: "Output", desc: "User gets Z" },
  ],
  
  // Slide 5: Tech Stack
  techStack: ["Next.js", "FastAPI", "OpenAI GPT-4", "Python", "Tailwind CSS"],
  casePartnerTech: "Built on Microsoft Azure AI",  // Mention case partner tech
  
  // Slide 6: Impact
  impactMetrics: [
    { value: "10x", label: "faster than manual process" },
    { value: "95%", label: "accuracy on test data" },
    { value: "€2M", label: "potential annual savings" },
  ],
  
  // Slide 7: Team
  team: [
    { name: "Person 1", role: "Tech Lead" },
    { name: "Person 2", role: "Full Stack" },
    { name: "Person 3", role: "ML Engineer" },
    { name: "Person 4", role: "Product & Pitch" },
  ],
  
  // Slide 8: Vision / What's Next
  visionStatement: "With more time, we'd scale this to...",
  nextSteps: [
    "Integrate with real [case partner] API",
    "Add [advanced feature]",
    "Launch pilot with [target users]",
  ],
};

// =============================================================
// 🎨 THEME — Modify colors if desired
// =============================================================

const THEME = {
  // Ocean Gradient palette — professional and tech-forward
  primary: "065A82",      // Deep blue
  secondary: "1C7293",    // Teal
  accent: "02C39A",       // Mint green
  dark: "0B132B",         // Near black
  light: "F8F9FA",        // Off white
  white: "FFFFFF",
  muted: "6C757D",        // Gray
  
  // Fonts
  headerFont: "Trebuchet MS",
  bodyFont: "Calibri",
};

// =============================================================
// 🏗️  SLIDE GENERATION — DO NOT MODIFY BELOW
// =============================================================

let pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = CONFIG.teamName;
pres.title = `${CONFIG.projectName} — Hackathon Pitch`;

// --- Helper Functions ---
function addDarkBg(slide) {
  slide.background = { color: THEME.dark };
}
function addLightBg(slide) {
  slide.background = { color: THEME.light };
}
function addFooter(slide, dark = false) {
  const color = dark ? "4A5568" : "8899AA";
  slide.addText(`${CONFIG.teamName}  |  ${CONFIG.casePartner}`, {
    x: 0.5, y: 5.1, w: 9, h: 0.4,
    fontSize: 9, color: color, fontFace: THEME.bodyFont
  });
}

// ── SLIDE 1: Title ──
{
  let slide = pres.addSlide();
  addDarkBg(slide);
  
  // Accent bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.12, h: 5.625, fill: { color: THEME.accent }
  });
  
  slide.addText(CONFIG.projectName.toUpperCase(), {
    x: 0.8, y: 1.2, w: 8, h: 1.2,
    fontSize: 44, fontFace: THEME.headerFont, color: THEME.white,
    bold: true, charSpacing: 3
  });
  
  slide.addText(CONFIG.tagline, {
    x: 0.8, y: 2.5, w: 7, h: 0.8,
    fontSize: 20, fontFace: THEME.bodyFont, color: THEME.accent,
    italic: true
  });
  
  slide.addText(`${CONFIG.teamName}  ·  ${CONFIG.casePartner} Challenge`, {
    x: 0.8, y: 4.2, w: 7, h: 0.5,
    fontSize: 14, fontFace: THEME.bodyFont, color: THEME.muted
  });
}

// ── SLIDE 2: Problem / Hook ──
{
  let slide = pres.addSlide();
  addDarkBg(slide);
  
  slide.addText("THE PROBLEM", {
    x: 0.8, y: 0.5, w: 8, h: 0.5,
    fontSize: 12, fontFace: THEME.bodyFont, color: THEME.accent,
    bold: true, charSpacing: 4
  });
  
  // Big stat
  slide.addText(CONFIG.hookStat, {
    x: 0.8, y: 1.2, w: 4, h: 2,
    fontSize: 72, fontFace: THEME.headerFont, color: THEME.accent,
    bold: true, margin: 0
  });
  
  slide.addText(CONFIG.hookStatLabel, {
    x: 0.8, y: 3.0, w: 5, h: 0.6,
    fontSize: 20, fontFace: THEME.bodyFont, color: THEME.white
  });
  
  slide.addText(CONFIG.problemDescription, {
    x: 0.8, y: 3.8, w: 8, h: 1,
    fontSize: 14, fontFace: THEME.bodyFont, color: THEME.muted
  });
  
  addFooter(slide, true);
}

// ── SLIDE 3: Solution ──
{
  let slide = pres.addSlide();
  addLightBg(slide);
  
  slide.addText("OUR SOLUTION", {
    x: 0.8, y: 0.5, w: 8, h: 0.5,
    fontSize: 12, fontFace: THEME.bodyFont, color: THEME.primary,
    bold: true, charSpacing: 4
  });
  
  slide.addText(CONFIG.solutionOneLiner, {
    x: 0.8, y: 1.2, w: 8.4, h: 1,
    fontSize: 22, fontFace: THEME.headerFont, color: THEME.dark,
    bold: true
  });
  
  // Feature cards
  const cardW = 2.6;
  const cardGap = 0.3;
  const startX = 0.8;
  
  CONFIG.keyFeatures.forEach((feat, i) => {
    const x = startX + i * (cardW + cardGap);
    
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: x, y: 2.6, w: cardW, h: 2.2,
      fill: { color: THEME.white }, rectRadius: 0.1,
      shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.08 }
    });
    
    // Number circle
    slide.addShape(pres.shapes.OVAL, {
      x: x + 0.15, y: 2.8, w: 0.45, h: 0.45,
      fill: { color: THEME.accent }
    });
    slide.addText(`${i + 1}`, {
      x: x + 0.15, y: 2.8, w: 0.45, h: 0.45,
      fontSize: 16, fontFace: THEME.headerFont, color: THEME.white,
      bold: true, align: "center", valign: "middle"
    });
    
    slide.addText(feat.title, {
      x: x + 0.15, y: 3.4, w: cardW - 0.3, h: 0.4,
      fontSize: 14, fontFace: THEME.headerFont, color: THEME.dark,
      bold: true
    });
    
    slide.addText(feat.desc, {
      x: x + 0.15, y: 3.8, w: cardW - 0.3, h: 0.8,
      fontSize: 11, fontFace: THEME.bodyFont, color: THEME.muted
    });
  });
  
  addFooter(slide);
}

// ── SLIDE 4: Demo (minimal — live demo happens here) ──
{
  let slide = pres.addSlide();
  addDarkBg(slide);
  
  slide.addText("LIVE DEMO", {
    x: 0.8, y: 1.5, w: 8.4, h: 1,
    fontSize: 44, fontFace: THEME.headerFont, color: THEME.white,
    bold: true, align: "center"
  });
  
  slide.addText(CONFIG.demoUrl, {
    x: 0.8, y: 2.8, w: 8.4, h: 0.6,
    fontSize: 18, fontFace: THEME.bodyFont, color: THEME.accent,
    align: "center"
  });
  
  slide.addText("[ Switch to browser for live demonstration ]", {
    x: 0.8, y: 3.8, w: 8.4, h: 0.5,
    fontSize: 12, fontFace: THEME.bodyFont, color: THEME.muted,
    italic: true, align: "center"
  });
}

// ── SLIDE 5: How It Works ──
{
  let slide = pres.addSlide();
  addLightBg(slide);
  
  slide.addText("HOW IT WORKS", {
    x: 0.8, y: 0.5, w: 8, h: 0.5,
    fontSize: 12, fontFace: THEME.bodyFont, color: THEME.primary,
    bold: true, charSpacing: 4
  });
  
  const stepW = 2.4;
  const arrowW = 0.6;
  const totalW = CONFIG.flowSteps.length * stepW + (CONFIG.flowSteps.length - 1) * arrowW;
  const flowStartX = (10 - totalW) / 2;
  
  CONFIG.flowSteps.forEach((step, i) => {
    const x = flowStartX + i * (stepW + arrowW);
    
    // Step circle
    slide.addShape(pres.shapes.OVAL, {
      x: x + (stepW - 1.2) / 2, y: 1.6, w: 1.2, h: 1.2,
      fill: { color: THEME.primary }
    });
    slide.addText(step.step, {
      x: x + (stepW - 1.2) / 2, y: 1.6, w: 1.2, h: 1.2,
      fontSize: 28, fontFace: THEME.headerFont, color: THEME.white,
      bold: true, align: "center", valign: "middle"
    });
    
    // Title + desc
    slide.addText(step.title, {
      x: x, y: 3.0, w: stepW, h: 0.5,
      fontSize: 16, fontFace: THEME.headerFont, color: THEME.dark,
      bold: true, align: "center"
    });
    slide.addText(step.desc, {
      x: x, y: 3.5, w: stepW, h: 0.8,
      fontSize: 11, fontFace: THEME.bodyFont, color: THEME.muted,
      align: "center"
    });
    
    // Arrow between steps
    if (i < CONFIG.flowSteps.length - 1) {
      slide.addText("→", {
        x: x + stepW, y: 1.8, w: arrowW, h: 0.8,
        fontSize: 28, color: THEME.accent, align: "center", valign: "middle",
        fontFace: THEME.bodyFont
      });
    }
  });
  
  addFooter(slide);
}

// ── SLIDE 6: Tech Stack ──
{
  let slide = pres.addSlide();
  addLightBg(slide);
  
  slide.addText("TECH STACK", {
    x: 0.8, y: 0.5, w: 8, h: 0.5,
    fontSize: 12, fontFace: THEME.bodyFont, color: THEME.primary,
    bold: true, charSpacing: 4
  });
  
  // Tech pills
  let pillX = 0.8;
  let pillY = 1.5;
  const pillH = 0.55;
  const pillGap = 0.25;
  
  CONFIG.techStack.forEach((tech) => {
    const pillW = tech.length * 0.13 + 0.6;
    
    if (pillX + pillW > 9.5) {
      pillX = 0.8;
      pillY += pillH + pillGap;
    }
    
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: pillX, y: pillY, w: pillW, h: pillH,
      fill: { color: THEME.primary }, rectRadius: 0.15
    });
    slide.addText(tech, {
      x: pillX, y: pillY, w: pillW, h: pillH,
      fontSize: 13, fontFace: THEME.bodyFont, color: THEME.white,
      align: "center", valign: "middle"
    });
    
    pillX += pillW + pillGap;
  });
  
  // Case partner callout
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.8, y: 3.5, w: 8.4, h: 1,
    fill: { color: THEME.white }, rectRadius: 0.1,
    line: { color: THEME.accent, width: 2 }
  });
  slide.addText(CONFIG.casePartnerTech, {
    x: 1.2, y: 3.5, w: 7.6, h: 1,
    fontSize: 16, fontFace: THEME.headerFont, color: THEME.primary,
    bold: true, valign: "middle"
  });
  
  addFooter(slide);
}

// ── SLIDE 7: Impact ──
{
  let slide = pres.addSlide();
  addDarkBg(slide);
  
  slide.addText("IMPACT", {
    x: 0.8, y: 0.5, w: 8, h: 0.5,
    fontSize: 12, fontFace: THEME.bodyFont, color: THEME.accent,
    bold: true, charSpacing: 4
  });
  
  const metricW = 2.8;
  const metricGap = 0.3;
  const metricStartX = (10 - (CONFIG.impactMetrics.length * metricW + (CONFIG.impactMetrics.length - 1) * metricGap)) / 2;
  
  CONFIG.impactMetrics.forEach((metric, i) => {
    const x = metricStartX + i * (metricW + metricGap);
    
    slide.addText(metric.value, {
      x: x, y: 1.5, w: metricW, h: 1.5,
      fontSize: 56, fontFace: THEME.headerFont, color: THEME.accent,
      bold: true, align: "center", valign: "middle"
    });
    
    slide.addText(metric.label, {
      x: x, y: 3.0, w: metricW, h: 0.8,
      fontSize: 14, fontFace: THEME.bodyFont, color: THEME.white,
      align: "center"
    });
  });
  
  addFooter(slide, true);
}

// ── SLIDE 8: Team ──
{
  let slide = pres.addSlide();
  addLightBg(slide);
  
  slide.addText("THE TEAM", {
    x: 0.8, y: 0.5, w: 8, h: 0.5,
    fontSize: 12, fontFace: THEME.bodyFont, color: THEME.primary,
    bold: true, charSpacing: 4
  });
  
  const memberW = 2;
  const memberGap = 0.4;
  const memberStartX = (10 - (CONFIG.team.length * memberW + (CONFIG.team.length - 1) * memberGap)) / 2;
  
  CONFIG.team.forEach((member, i) => {
    const x = memberStartX + i * (memberW + memberGap);
    
    // Avatar circle placeholder
    slide.addShape(pres.shapes.OVAL, {
      x: x + (memberW - 1) / 2, y: 1.5, w: 1, h: 1,
      fill: { color: THEME.secondary }
    });
    slide.addText(member.name.charAt(0).toUpperCase(), {
      x: x + (memberW - 1) / 2, y: 1.5, w: 1, h: 1,
      fontSize: 28, fontFace: THEME.headerFont, color: THEME.white,
      bold: true, align: "center", valign: "middle"
    });
    
    slide.addText(member.name, {
      x: x, y: 2.7, w: memberW, h: 0.4,
      fontSize: 14, fontFace: THEME.headerFont, color: THEME.dark,
      bold: true, align: "center"
    });
    slide.addText(member.role, {
      x: x, y: 3.1, w: memberW, h: 0.4,
      fontSize: 11, fontFace: THEME.bodyFont, color: THEME.muted,
      align: "center"
    });
  });
  
  addFooter(slide);
}

// ── SLIDE 9: Vision / What's Next ──
{
  let slide = pres.addSlide();
  addDarkBg(slide);
  
  // Accent bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.12, h: 5.625, fill: { color: THEME.accent }
  });
  
  slide.addText("WHAT'S NEXT", {
    x: 0.8, y: 0.5, w: 8, h: 0.5,
    fontSize: 12, fontFace: THEME.bodyFont, color: THEME.accent,
    bold: true, charSpacing: 4
  });
  
  slide.addText(CONFIG.visionStatement, {
    x: 0.8, y: 1.3, w: 8, h: 0.8,
    fontSize: 22, fontFace: THEME.headerFont, color: THEME.white,
    bold: true
  });
  
  const nextStepItems = CONFIG.nextSteps.map((step, i) => ({
    text: step,
    options: { bullet: true, breakLine: i < CONFIG.nextSteps.length - 1, color: THEME.light }
  }));
  
  slide.addText(nextStepItems, {
    x: 0.8, y: 2.4, w: 8, h: 2,
    fontSize: 16, fontFace: THEME.bodyFont, color: THEME.light,
    lineSpacingMultiple: 1.5
  });
  
  // Closing CTA
  slide.addText(`Thank you! — ${CONFIG.teamName}`, {
    x: 0.8, y: 4.5, w: 8.4, h: 0.5,
    fontSize: 16, fontFace: THEME.headerFont, color: THEME.accent,
    bold: true, align: "center"
  });
}

// ── Generate ──
const outputPath = "hackathon_pitch_deck.pptx";
pres.writeFile({ fileName: outputPath }).then(() => {
  console.log(`✅ Pitch deck saved to ${outputPath}`);
  console.log(`   Slides: 9`);
  console.log(`   Edit the CONFIG section to customize content.`);
});

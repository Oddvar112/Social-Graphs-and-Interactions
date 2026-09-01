/* ===========================================================================
   hero.js: the three playable characters, drawn by hand on canvas.

   No image files: each hero is a small articulated skeleton (hips, knees,
   shoulders, elbows, head) that an animation function poses and a costume
   function then dresses. That lets all three share one gait engine while moving
   completely differently: Spider-Man swings, Wolverine sprints, Strange floats.

   Coordinate system: the origin sits under the feet and y points DOWN (canvas
   convention), so the whole figure has negative y. Full height ~41 units, so
   ctx.scale(s, s) makes the figure 41*s pixels tall.
   =========================================================================== */

(function (global) {
  'use strict';

  // ---- skeleton metrics --------------------------------------------------
  const HIP_Y = -18, SH_Y = -29, HEAD_Y = -36.5, HEAD_R = 5.2;
  const THIGH = 9, SHIN = 9, UPPER = 7, FORE = 7;
  const HIP_X = 3.1, SH_X = 5.2;

  const TAU = Math.PI * 2;
  const lerp = (a, b, t) => a + (b - a) * t;

  function rot(p, cx, cy, a) {
    const s = Math.sin(a), c = Math.cos(a), dx = p.x - cx, dy = p.y - cy;
    return { x: cx + dx * c - dy * s, y: cy + dx * s + dy * c };
  }

  /* Builds every joint position from a pose.
     Angles are 0 = straight down, positive = forward. The upper body rotates
     about the hips by `lean`; the legs hang from the unrotated hips. */
  function skeleton(pose) {
    const p = Object.assign({
      lean: 0, bob: 0, headTilt: 0,
      legA: [0, 0], legB: [0, 0],     // [thigh, knee delta]
      armA: [0, 0], armB: [0, 0],     // [upper arm, elbow delta]
    }, pose);

    const bob = p.bob;
    const hipC = { x: 0, y: HIP_Y + bob };

    function limb(ax, ay, a1, a2, l1, l2) {
      const j = { x: ax + l1 * Math.sin(a1), y: ay + l1 * Math.cos(a1) };
      const e = { x: j.x + l2 * Math.sin(a1 + a2), y: j.y + l2 * Math.cos(a1 + a2) };
      return { root: { x: ax, y: ay }, mid: j, tip: e };
    }

    const legs = [
      limb(-HIP_X, hipC.y, p.legA[0], p.legA[1], THIGH, SHIN),
      limb(HIP_X, hipC.y, p.legB[0], p.legB[1], THIGH, SHIN),
    ];

    // Upper body: build it upright, then rotate everything about the hip centre.
    const R = (pt) => rot(pt, hipC.x, hipC.y, p.lean);
    const shL = R({ x: -SH_X, y: SH_Y + bob });
    const shR = R({ x: SH_X, y: SH_Y + bob });
    const neck = R({ x: 0, y: SH_Y - 1.5 + bob });
    const head = R({ x: 0, y: HEAD_Y + bob });

    const arms = [
      limb(shL.x, shL.y, p.armA[0] + p.lean, p.armA[1], UPPER, FORE),
      limb(shR.x, shR.y, p.armB[0] + p.lean, p.armB[1], UPPER, FORE),
    ];

    return { hipC, legs, arms, shL, shR, neck, head, lean: p.lean, tilt: p.headTilt };
  }

  // ---- pose generators ---------------------------------------------------
  // Each returns a pose for time t (seconds). `run` is the shared engine.

  function poseRun(t, speed) {
    const p = t * speed;
    const sw = Math.sin(p), sw2 = Math.sin(p + Math.PI);
    return {
      lean: 0.30,
      bob: -Math.abs(Math.sin(p)) * 1.6,
      legA: [sw * 0.95 - 0.15, Math.max(0, -sw) * 1.5 + 0.15],
      legB: [sw2 * 0.95 - 0.15, Math.max(0, -sw2) * 1.5 + 0.15],
      armA: [sw2 * 1.25 + 0.15, -1.5],
      armB: [sw * 1.25 + 0.15, -1.5],
    };
  }

  function poseIdleStand(t) {
    const b = Math.sin(t * 1.7);
    return {
      lean: 0.04 + b * 0.02,
      bob: b * 0.5,
      legA: [-0.06, 0.06], legB: [0.06, 0.06],
      armA: [0.10 + b * 0.06, -0.30], armB: [-0.10 - b * 0.06, -0.30],
    };
  }

  // Spider-Man crouches and scans the skyline when idle.
  function poseCrouch(t) {
    const b = Math.sin(t * 2.1);
    return {
      lean: 0.62 + b * 0.03,
      bob: 2.6 + b * 0.35,
      legA: [-0.95, 1.75], legB: [0.85, 1.65],
      armA: [-0.55, -0.55], armB: [0.75, -0.95],
    };
  }

  // Spider-Man mid-swing: body stretched out, one hand gripping the web line.
  function poseSwing(t, prog) {
    const s = Math.sin(t * 6);
    const tuck = Math.sin(Math.min(1, prog) * Math.PI);   // tucks at the apex
    return {
      lean: -0.55 + Math.sin(prog * TAU) * 0.35,
      bob: 0,
      legA: [-0.30 - tuck * 1.5, 0.25 + tuck * 1.9],
      legB: [0.55 + tuck * 0.7, 0.20 + tuck * 1.2],
      armA: [-2.55 + s * 0.05, -0.15],                     // the web-line grip
      armB: [1.25 + s * 0.12, -0.45],
    };
  }

  // Doctor Strange floats cross-legged while his hands trace mystic circles.
  function poseFloat(t, moving) {
    const b = Math.sin(t * 1.9), c = Math.cos(t * 2.6);
    return {
      lean: moving ? 0.16 : 0.02,
      bob: -2.4 + b * 0.9,
      legA: [-0.55, 1.15 + b * 0.05], legB: [0.50, 1.05 - b * 0.05],
      armA: [-1.15 + c * 0.16, -0.85], armB: [1.05 - c * 0.16, -0.75],
    };
  }

  // ---- drawing helpers ---------------------------------------------------
  function bone(ctx, a, b, w, color) {
    ctx.strokeStyle = color; ctx.lineWidth = w; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }

  function torso(ctx, sk, color) {
    const { shL, shR, hipC, lean } = sk;
    const hL = rot({ x: -HIP_X - 0.7, y: hipC.y + 1 }, hipC.x, hipC.y, lean);
    const hR = rot({ x: HIP_X + 0.7, y: hipC.y + 1 }, hipC.x, hipC.y, lean);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(shL.x - 1.4, shL.y - 1.2);
    ctx.quadraticCurveTo(0, shL.y - 3.2, shR.x + 1.4, shR.y - 1.2);
    ctx.lineTo(hR.x, hR.y);
    ctx.quadraticCurveTo(0, hipC.y + 2.4, hL.x, hL.y);
    ctx.closePath(); ctx.fill();
  }

  function head(ctx, sk, color) {
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.ellipse(sk.head.x, sk.head.y, HEAD_R * 0.92, HEAD_R, sk.lean, 0, TAU);
    ctx.fill();
  }

  // ---- costumes ----------------------------------------------------------
  // Each costume function receives the skeleton and paints back-to-front.

  function suitSpidey(ctx, sk, c, facing) {
    // far-side limbs
    bone(ctx, sk.arms[0].root, sk.arms[0].mid, 3.4, c.armDark);
    bone(ctx, sk.arms[0].mid, sk.arms[0].tip, 3.0, c.armDark);
    bone(ctx, sk.legs[0].root, sk.legs[0].mid, 4.2, c.legDark);
    bone(ctx, sk.legs[0].mid, sk.legs[0].tip, 3.6, c.legDark);

    torso(ctx, sk, c.suit);

    // chest spider
    ctx.save();
    ctx.translate(sk.shL.x + (sk.shR.x - sk.shL.x) / 2, (sk.shL.y + sk.hipC.y) / 2);
    ctx.rotate(sk.lean); ctx.fillStyle = c.ink;
    ctx.beginPath(); ctx.ellipse(0, 0, 1.5, 2.2, 0, 0, TAU); ctx.fill();
    ctx.lineWidth = 0.55; ctx.strokeStyle = c.ink;
    for (let i = 0; i < 4; i++) {
      const a = 0.5 + i * 0.42;
      ctx.beginPath();
      ctx.moveTo(-1.2, -1.4 + i * 0.9); ctx.lineTo(-3.4, -2.4 + i * 1.5);
      ctx.moveTo(1.2, -1.4 + i * 0.9); ctx.lineTo(3.4, -2.4 + i * 1.5);
      ctx.stroke();
    }
    ctx.restore();

    // near-side limbs
    bone(ctx, sk.legs[1].root, sk.legs[1].mid, 4.4, c.leg);
    bone(ctx, sk.legs[1].mid, sk.legs[1].tip, 3.8, c.leg);
    bone(ctx, sk.arms[1].root, sk.arms[1].mid, 3.6, c.arm);
    bone(ctx, sk.arms[1].mid, sk.arms[1].tip, 3.2, c.arm);

    head(ctx, sk, c.suit);

    // mask: web pattern plus the big white eyes
    ctx.save();
    ctx.translate(sk.head.x, sk.head.y); ctx.rotate(sk.lean); ctx.scale(facing, 1);
    ctx.strokeStyle = 'rgba(0,0,0,.34)'; ctx.lineWidth = 0.35;
    for (let i = -1; i <= 2; i++) {
      ctx.beginPath(); ctx.arc(0, -1.2, 1.9 + i * 1.35, 0.15, Math.PI - 0.15); ctx.stroke();
    }
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.moveTo(0.4, -1.5); ctx.quadraticCurveTo(3.6, -2.6, 4.0, 0.1);
    ctx.quadraticCurveTo(2.4, 1.2, 0.5, 0.2); ctx.closePath(); ctx.fill();
    ctx.beginPath();
    ctx.moveTo(-0.4, -1.5); ctx.quadraticCurveTo(-2.6, -2.3, -3.0, 0.0);
    ctx.quadraticCurveTo(-1.8, 0.9, -0.5, 0.2); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = c.ink; ctx.lineWidth = 0.45; ctx.stroke();
    ctx.restore();
  }

  function suitWolverine(ctx, sk, c, facing, opts) {
    bone(ctx, sk.arms[0].root, sk.arms[0].mid, 3.4, c.armDark);
    bone(ctx, sk.arms[0].mid, sk.arms[0].tip, 3.0, c.armDark);
    bone(ctx, sk.legs[0].root, sk.legs[0].mid, 4.2, c.legDark);
    bone(ctx, sk.legs[0].mid, sk.legs[0].tip, 3.6, c.legDark);

    torso(ctx, sk, c.suit);
    // blue side panels and belt
    ctx.save();
    ctx.translate(0, 0);
    ctx.strokeStyle = c.arm; ctx.lineWidth = 1.6; ctx.lineCap = 'butt';
    ctx.beginPath();
    ctx.moveTo(sk.shL.x - 0.4, sk.shL.y); ctx.lineTo(sk.hipC.x - 2.4, sk.hipC.y + 1);
    ctx.moveTo(sk.shR.x + 0.4, sk.shR.y); ctx.lineTo(sk.hipC.x + 2.4, sk.hipC.y + 1);
    ctx.stroke();
    ctx.fillStyle = c.belt;
    const bl = rot({ x: -4.6, y: sk.hipC.y - 0.4 }, sk.hipC.x, sk.hipC.y, sk.lean);
    ctx.save(); ctx.translate(bl.x, bl.y); ctx.rotate(sk.lean);
    ctx.fillRect(0, 0, 9.2, 2.1); ctx.restore();
    ctx.restore();

    bone(ctx, sk.legs[1].root, sk.legs[1].mid, 4.4, c.leg);
    bone(ctx, sk.legs[1].mid, sk.legs[1].tip, 3.8, c.leg);
    bone(ctx, sk.arms[1].root, sk.arms[1].mid, 3.6, c.arm);
    bone(ctx, sk.arms[1].mid, sk.arms[1].tip, 3.2, c.arm);

    // claws: out while he is moving
    if (opts.claws > 0.01) {
      ctx.save();
      ctx.strokeStyle = '#eaf2ff'; ctx.lineCap = 'round'; ctx.lineWidth = 0.75;
      ctx.shadowColor = '#bfe8ff'; ctx.shadowBlur = 3;
      for (const arm of sk.arms) {
        const dx = arm.tip.x - arm.mid.x, dy = arm.tip.y - arm.mid.y;
        const L = Math.hypot(dx, dy) || 1, ux = dx / L, uy = dy / L;
        for (let k = -1; k <= 1; k++) {
          const px = -uy * k * 1.15, py = ux * k * 1.15;
          ctx.beginPath();
          ctx.moveTo(arm.tip.x + px, arm.tip.y + py);
          ctx.lineTo(arm.tip.x + px + ux * 7.5 * opts.claws, arm.tip.y + py + uy * 7.5 * opts.claws);
          ctx.stroke();
        }
      }
      ctx.restore();
    }

    head(ctx, sk, c.suit);
    // the cowl with its two points
    ctx.save();
    ctx.translate(sk.head.x, sk.head.y); ctx.rotate(sk.lean); ctx.scale(facing, 1);
    ctx.fillStyle = c.suit;
    ctx.beginPath();
    ctx.moveTo(-4.4, -2.2); ctx.lineTo(-6.6, -7.4); ctx.lineTo(-2.2, -4.2); ctx.closePath(); ctx.fill();
    ctx.beginPath();
    ctx.moveTo(4.4, -2.2); ctx.lineTo(6.6, -7.4); ctx.lineTo(2.2, -4.2); ctx.closePath(); ctx.fill();
    ctx.fillStyle = c.ink;                                   // mask opening
    ctx.beginPath();
    ctx.moveTo(-3.4, 0.6); ctx.quadraticCurveTo(0, 2.9, 3.4, 0.6);
    ctx.quadraticCurveTo(0, 1.5, -3.4, 0.6); ctx.closePath(); ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.moveTo(0.7, -1.6); ctx.lineTo(3.5, -1.0); ctx.lineTo(1.0, 0.0); ctx.closePath(); ctx.fill();
    ctx.beginPath(); ctx.moveTo(-0.7, -1.6); ctx.lineTo(-3.2, -1.0); ctx.lineTo(-1.0, 0.0); ctx.closePath(); ctx.fill();
    ctx.restore();
  }

  function suitStrange(ctx, sk, c, facing, opts) {
    // cloak: a rippling polygon behind the body
    ctx.save();
    const w = opts.time * 3.4, sway = opts.moving ? 3.2 : 1.3;
    ctx.fillStyle = c.cape;
    ctx.beginPath();
    ctx.moveTo(sk.shL.x - 0.5, sk.shL.y - 1);
    ctx.lineTo(sk.shR.x + 0.5, sk.shR.y - 1);
    for (let i = 1; i <= 6; i++) {
      const f = i / 6;
      const y = lerp(sk.shR.y, sk.hipC.y + 15, f);
      const x = sk.shR.x + 1.5 + Math.sin(w + f * 3.1) * sway * f * 1.6 - facing * f * 5.5;
      ctx.lineTo(x, y + Math.sin(w * 1.3 + f * 2) * f * 1.4);
    }
    for (let i = 6; i >= 1; i--) {
      const f = i / 6;
      const y = lerp(sk.shL.y, sk.hipC.y + 13, f);
      const x = sk.shL.x - 1.5 + Math.sin(w + f * 3.1 + 1.2) * sway * f * 1.3 - facing * f * 7.5;
      ctx.lineTo(x, y + Math.sin(w * 1.3 + f * 2 + 1) * f * 1.2);
    }
    ctx.closePath(); ctx.fill();
    ctx.strokeStyle = c.capeEdge; ctx.lineWidth = 0.7; ctx.stroke();
    ctx.restore();

    bone(ctx, sk.arms[0].root, sk.arms[0].mid, 3.4, c.armDark);
    bone(ctx, sk.arms[0].mid, sk.arms[0].tip, 3.0, c.armDark);
    bone(ctx, sk.legs[0].root, sk.legs[0].mid, 4.2, c.legDark);
    bone(ctx, sk.legs[0].mid, sk.legs[0].tip, 3.6, c.legDark);

    torso(ctx, sk, c.suit);
    ctx.strokeStyle = c.belt; ctx.lineWidth = 1.5;           // the golden sash
    const b1 = rot({ x: -4.4, y: sk.hipC.y - 1.2 }, sk.hipC.x, sk.hipC.y, sk.lean);
    const b2 = rot({ x: 4.4, y: sk.hipC.y - 1.2 }, sk.hipC.x, sk.hipC.y, sk.lean);
    ctx.beginPath(); ctx.moveTo(b1.x, b1.y); ctx.lineTo(b2.x, b2.y); ctx.stroke();

    bone(ctx, sk.legs[1].root, sk.legs[1].mid, 4.4, c.leg);
    bone(ctx, sk.legs[1].mid, sk.legs[1].tip, 3.8, c.leg);
    bone(ctx, sk.arms[1].root, sk.arms[1].mid, 3.6, c.arm);
    bone(ctx, sk.arms[1].mid, sk.arms[1].tip, 3.2, c.arm);

    // mystic rune rings in his hands
    ctx.save();
    ctx.strokeStyle = c.magic; ctx.shadowColor = c.magic;
    ctx.shadowBlur = 6; ctx.lineWidth = 0.6;
    for (const arm of sk.arms) {
      const r = 2.6 + Math.sin(opts.time * 4) * 0.35;
      ctx.beginPath(); ctx.ellipse(arm.tip.x, arm.tip.y, r, r * 0.42, opts.time * 1.6, 0, TAU); ctx.stroke();
      ctx.beginPath(); ctx.ellipse(arm.tip.x, arm.tip.y, r * 0.55, r * 0.24, -opts.time * 2.2, 0, TAU); ctx.stroke();
    }
    ctx.restore();

    head(ctx, sk, c.skin);
    ctx.save();
    ctx.translate(sk.head.x, sk.head.y); ctx.rotate(sk.lean); ctx.scale(facing, 1);
    ctx.fillStyle = c.hair;                                   // hair and temples
    ctx.beginPath(); ctx.arc(0, -1.1, HEAD_R * 0.95, Math.PI, TAU); ctx.fill();
    ctx.fillStyle = '#e8ecf7';
    ctx.beginPath(); ctx.ellipse(-4.1, -0.9, 0.9, 1.7, 0.2, 0, TAU); ctx.fill();
    ctx.beginPath(); ctx.ellipse(4.1, -0.9, 0.9, 1.7, -0.2, 0, TAU); ctx.fill();
    ctx.fillStyle = c.hair;                                   // goatee
    ctx.beginPath(); ctx.ellipse(0.4, 3.5, 1.5, 1.9, 0, 0, TAU); ctx.fill();
    ctx.fillStyle = c.ink;                                    // eyes
    ctx.beginPath(); ctx.ellipse(1.9, 0.2, 0.75, 0.55, 0, 0, TAU); ctx.fill();
    ctx.beginPath(); ctx.ellipse(-1.5, 0.2, 0.65, 0.5, 0, 0, TAU); ctx.fill();
    ctx.restore();
  }

  // ---- hero specs --------------------------------------------------------
  const HEROES = [
    {
      key: 'spidey',
      node: 'Spider-Man',
      label: 'SPIDER-MAN',
      role: 'The sun the graph orbits',
      blurb: '106 pages link to him, a third of the whole universe. Almost every road leads home. But he only links out to 9 places himself.',
      tint: '#e62429',
      travel: 'swing',
      speed: 1.0,
      colors: { suit: '#d42b2f', arm: '#2f5bd4', armDark: '#22429b', leg: '#2f5bd4', legDark: '#22429b', ink: '#0b0f1c' },
    },
    {
      key: 'logan',
      node: 'Wolverine_(character)',
      label: 'WOLVERINE',
      role: 'Right in the thick of it',
      blurb: '60 in, 17 out. He sits where the graph is densest, so there are plenty of routes home and plenty of ways to get lost next door.',
      tint: '#f2c521',
      travel: 'run',
      speed: 1.15,
      colors: { suit: '#f2c521', arm: '#2f5bd4', armDark: '#22429b', leg: '#f2c521', legDark: '#c39d15', ink: '#0b0f1c', belt: '#8b4a12' },
    },
    {
      key: 'strange',
      node: 'Doctor_Strange',
      label: 'DR. STRANGE',
      role: 'The hard way home',
      blurb: '50 links in and good reach outward, but his neighbourhood is strange: it is easy to end up in a corner that never points back.',
      tint: '#a678ff',
      travel: 'float',
      speed: 0.85,
      colors: {
        suit: '#2a3f8f', arm: '#2a3f8f', armDark: '#1d2c66', leg: '#1b2450', legDark: '#141b3c',
        ink: '#0b0f1c', belt: '#e8b53a', cape: '#b8232b', capeEdge: '#7d1219',
        skin: '#e3b291', hair: '#1d1f2b', magic: '#ffb03a',
      },
    },
  ];

  /* Main entry point. ctx must already be translated so that (0,0) sits under
     the feet, and scaled to the desired size. */
  function drawHero(ctx, hero, opts) {
    const o = Object.assign({ time: 0, facing: 1, moving: false, progress: 0 }, opts);
    const t = o.time;
    let pose;

    if (hero.travel === 'swing') {
      pose = o.moving ? poseSwing(t, o.progress) : poseCrouch(t);
    } else if (hero.travel === 'run') {
      pose = o.moving ? poseRun(t, 12 * hero.speed) : poseIdleStand(t);
    } else {
      pose = poseFloat(t, o.moving);
    }

    const sk = skeleton(pose);
    ctx.save();
    ctx.scale(o.facing, 1);          // mirror the whole figure to face travel direction
    const facing = 1;

    // ground shadow
    if (!o.moving && hero.travel !== 'float') {
      ctx.fillStyle = 'rgba(0,0,0,.35)';
      ctx.beginPath(); ctx.ellipse(0, 1.2, 7.5, 1.8, 0, 0, TAU); ctx.fill();
    }

    if (hero.travel === 'swing') suitSpidey(ctx, sk, hero.colors, facing);
    else if (hero.travel === 'run') suitWolverine(ctx, sk, hero.colors, facing, { claws: o.moving ? 1 : 0.25 });
    else suitStrange(ctx, sk, hero.colors, facing, { time: t, moving: o.moving });

    ctx.restore();
    return sk;
  }

  /* Where is the hand gripping the web? Used to draw Spider-Man's web line
     from the hero up to its anchor point. */
  function webHand(hero, time, progress) {
    const sk = skeleton(poseSwing(time, progress));
    return sk.arms[0].tip;
  }

  global.HEROES = HEROES;
  global.drawHero = drawHero;
  global.webHand = webHand;
})(window);

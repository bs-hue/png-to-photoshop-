/**
 * ag-psd compiler script
 * Reads manifest.json and layer assets, then outputs a PSD file with NATIVE EDITABLE TYPE LAYERS!
 * Fits font size precisely using canvas metrics so text doesn't blow out bounding boxes.
 * Usage: node build_psd.js <job_dir>
 */
const fs = require('fs');
const path = require('path');
const agPsd = require('ag-psd');
const { createCanvas, loadImage, registerFont } = require('canvas');

agPsd.initializeCanvas(createCanvas);

// Register downloaded font families so node-canvas renders pixel-perfect previews
const fontsDir = path.join(__dirname, 'fonts');
if (fs.existsSync(fontsDir)) {
  const tryReg = (file, family, weight, style) => {
    const full = path.join(fontsDir, file);
    if (fs.existsSync(full)) {
      try {
        registerFont(full, { family, weight, style });
      } catch (e) {}
    }
  };

  // Poppins
  tryReg('Poppins-Regular.ttf', 'Poppins', 'normal', 'normal');
  tryReg('Poppins-Medium.ttf', 'Poppins', '500', 'normal');
  tryReg('Poppins-SemiBold.ttf', 'Poppins', '600', 'normal');
  tryReg('Poppins-Bold.ttf', 'Poppins', 'bold', 'normal');
  tryReg('Poppins-Black.ttf', 'Poppins', '900', 'normal');
  // Playfair Display
  tryReg('PlayfairDisplay.ttf', 'Playfair Display', 'normal', 'normal');
  tryReg('PlayfairDisplay-Bold.ttf', 'Playfair Display', 'bold', 'normal');
  tryReg('PlayfairDisplay-Black.ttf', 'Playfair Display', '900', 'normal');
  tryReg('PlayfairDisplay-Italic.ttf', 'Playfair Display', 'normal', 'italic');
  tryReg('PlayfairDisplay-BoldItalic.ttf', 'Playfair Display', 'bold', 'italic');
  // Montserrat
  tryReg('Montserrat.ttf', 'Montserrat', 'normal', 'normal');
  tryReg('Montserrat-Regular.ttf', 'Montserrat', 'normal', 'normal');
  tryReg('Montserrat-Medium.ttf', 'Montserrat', '500', 'normal');
  tryReg('Montserrat-SemiBold.ttf', 'Montserrat', '600', 'normal');
  tryReg('Montserrat-Bold.ttf', 'Montserrat', 'bold', 'normal');
  tryReg('Montserrat-Black.ttf', 'Montserrat', '900', 'normal');
  tryReg('Montserrat-Italic.ttf', 'Montserrat', 'normal', 'italic');
  // Mukta (Devanagari)
  tryReg('Mukta-Regular.ttf', 'Mukta', 'normal', 'normal');
  tryReg('Mukta-Medium.ttf', 'Mukta', '500', 'normal');
  tryReg('Mukta-SemiBold.ttf', 'Mukta', '600', 'normal');
  tryReg('Mukta-Bold.ttf', 'Mukta', 'bold', 'normal');
  tryReg('Mukta-ExtraBold.ttf', 'Mukta', '900', 'normal');
  // Khand (Condensed Devanagari)
  tryReg('Khand-Regular.ttf', 'Khand', 'normal', 'normal');
  tryReg('Khand-Medium.ttf', 'Khand', '500', 'normal');
  tryReg('Khand-SemiBold.ttf', 'Khand', '600', 'normal');
  tryReg('Khand-Bold.ttf', 'Khand', 'bold', 'normal');
  // Rozha One (Serif Devanagari)
  tryReg('RozhaOne-Regular.ttf', 'Rozha One', 'normal', 'normal');
  // Modern Tech Sans
  tryReg('Euclid Circular B Regular.ttf', 'Euclid Circular B', 'normal', 'normal');
  tryReg('Euclid Circular B Bold.ttf', 'Euclid Circular B', 'bold', 'normal');
  tryReg('Manrope-Regular.ttf', 'Manrope', 'normal', 'normal');
  tryReg('Manrope-Bold.ttf', 'Manrope', 'bold', 'normal');
  tryReg('Sora-Regular.ttf', 'Sora', 'normal', 'normal');
  tryReg('Sora-Bold.ttf', 'Sora', 'bold', 'normal');
}

function calibrateFontMetrics(fontFamily, text, targetW, targetH, top) {
  if (!text || targetW <= 0) {
    const sz = Math.max(8, Math.round(targetH * 0.72));
    return { fontSize: sz, baselineY: Math.round(top + targetH) };
  }
  
  const c = createCanvas(3000, 800);
  const ctx = c.getContext('2d');
  
  let canvasFont = 'Poppins';
  if (/playfair/i.test(fontFamily)) canvasFont = 'Playfair Display';
  else if (/montserrat/i.test(fontFamily)) canvasFont = 'Montserrat';
  else if (/mukta/i.test(fontFamily)) canvasFont = 'Mukta';
  else if (/khand/i.test(fontFamily)) canvasFont = 'Khand';
  else if (/impact|bebas/i.test(fontFamily)) canvasFont = 'Impact';
  else if (/georgia|bodoni/i.test(fontFamily)) canvasFont = 'Georgia';
  else if (/times/i.test(fontFamily)) canvasFont = 'Times New Roman';
  else if (/nirmala/i.test(fontFamily)) canvasFont = 'Nirmala UI';
  else canvasFont = 'Poppins';

  const isBlack = /black|extrabold/i.test(fontFamily);
  const isBold = /bold/i.test(fontFamily) || isBlack;
  const isSemiBold = /semibold/i.test(fontFamily);
  const isMedium = /medium/i.test(fontFamily);
  const isItalic = /italic/i.test(fontFamily);
  const weight = isBlack ? '900' : (isBold ? 'bold' : (isSemiBold ? '600' : (isMedium ? '500' : 'normal')));
  const style = isItalic ? 'italic' : 'normal';

  const hasDevanagari = /[\u0900-\u097F]/.test(text);
  const hasDescenders = /[gjpqy]/.test(text);
  const isAllUpper = !/[a-z]/.test(text) && /[A-Z]/.test(text);
  let size;
  if (hasDevanagari) {
    size = targetH / 0.85;
  } else if (isAllUpper) {
    size = targetH / 0.72;
  } else if (hasDescenders) {
    size = targetH / 0.95;
  } else {
    size = targetH / 0.78;
  }

  // Test width at this natural font size
  ctx.font = `${style} ${weight} ${size}px "${canvasFont}"`;
  let m = ctx.measureText(text);
  
  if (m.width > 0) {
    const scaleFactor = targetW / m.width;
    if (scaleFactor < 0.96) {
      size = size * scaleFactor;
    } else if (scaleFactor > 1.05 && (size * scaleFactor * 0.72) <= targetH * 1.15) {
      size = size * Math.min(1.20, scaleFactor);
    }
  }
  
  // Clamp within reasonable typographic bounds
  size = Math.max(8, Math.min(targetH * 1.5, size));
  size = Math.round(size * 10) / 10;
  
  // Baseline calculation:
  let baselineY = hasDevanagari ? (top + Math.round(targetH * 0.82)) : (hasDescenders ? (top + targetH - Math.round(size * 0.18)) : (top + targetH));
  
  return { fontSize: size, baselineY: Math.round(baselineY) };
}

async function main() {
  const jobDir = process.argv[2];
  if (!jobDir) {
    console.error('Usage: node build_psd.js <job_dir>');
    process.exit(1);
  }

  const manifestPath = path.join(jobDir, 'manifest.json');
  if (!fs.existsSync(manifestPath)) {
    console.error('manifest.json not found in', jobDir);
    process.exit(1);
  }

  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  const width = manifest.width;
  const height = manifest.height;

  function hexToRgb(hex) {
    const n = parseInt(String(hex).replace('#', ''), 16);
    return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
  }

  // Map layer_id to layer info
  const layerMap = {};
  for (const l of manifest.layers) {
    layerMap[l.id] = l;
  }

  async function convertNode(node) {
    if (node.type === 'document') {
      const children = [];
      for (const child of node.children) {
        const c = await convertNode(child);
        if (c) children.push(c);
      }
      return children;
    }

    if (node.type === 'group') {
      const children = [];
      for (const child of node.children) {
        const c = await convertNode(child);
        if (c) children.push(c);
      }
      return {
        name: node.name,
        opened: true,
        children: children
      };
    }

    // Leaf layer
    const info = layerMap[node.layer_id || node.id];
    if (!info) return null;

    // IF THIS IS A TEXT LAYER: EMIT AS NATIVE PHOTOSHOP EDITABLE TYPE LAYER!
    if (info.type === 'text') {
      const t = info.text || {};
      const content = String(t.content || info.name || '').trim();
      const isDummy = !content || /^text region \d+$/i.test(content) || t.editable === false;
      
      if (!isDummy) {
        const b = info.bounds || { left: 0, top: 0, width: 100, height: 30 };
        const rgb = hexToRgb(t.color_hex || '#000000');
        const rawFont = t.font_family || 'ArialMT';
        
        // Map to exact PostScript font names recognized by Adobe Photoshop
        let psFontName = 'Poppins-Regular';
        let canvasFont = 'Poppins';
        const isBlack = /black|extrabold/i.test(rawFont);
        const isBold = /bold/i.test(rawFont) || isBlack;
        const isSemiBold = /semibold/i.test(rawFont);
        const isMedium = /medium/i.test(rawFont);
        const isItalic = /italic/i.test(rawFont);

        if (/playfair/i.test(rawFont)) {
          canvasFont = 'Playfair Display';
          if (isItalic) psFontName = isBold ? 'PlayfairDisplay-BoldItalic' : 'PlayfairDisplay-Italic';
          else if (isBlack) psFontName = 'PlayfairDisplay-Black';
          else if (isBold) psFontName = 'PlayfairDisplay-Bold';
          else psFontName = 'PlayfairDisplay-Regular';
        } else if (/montserrat/i.test(rawFont)) {
          canvasFont = 'Montserrat';
          if (isBlack) psFontName = 'Montserrat-Black';
          else if (isBold) psFontName = 'Montserrat-Bold';
          else if (isSemiBold) psFontName = 'Montserrat-SemiBold';
          else if (isMedium) psFontName = 'Montserrat-Medium';
          else psFontName = 'Montserrat-Regular';
        } else if (/mukta/i.test(rawFont)) {
          canvasFont = 'Mukta';
          if (isBlack) psFontName = 'Mukta-ExtraBold';
          else if (isBold) psFontName = 'Mukta-Bold';
          else if (isSemiBold) psFontName = 'Mukta-SemiBold';
          else if (isMedium) psFontName = 'Mukta-Medium';
          else psFontName = 'Mukta-Regular';
        } else if (/khand/i.test(rawFont)) {
          canvasFont = 'Khand';
          if (isBold || isBlack) psFontName = 'Khand-Bold';
          else if (isSemiBold) psFontName = 'Khand-SemiBold';
          else if (isMedium) psFontName = 'Khand-Medium';
          else psFontName = 'Khand-Regular';
        } else if (/rozha/i.test(rawFont)) {
          canvasFont = 'Rozha One';
          psFontName = 'RozhaOne-Regular';
        } else if (/euclid/i.test(rawFont)) {
          canvasFont = 'Euclid Circular B';
          if (isBold || isBlack) psFontName = 'EuclidCircularB-Bold';
          else if (isSemiBold) psFontName = 'EuclidCircularB-SemiBold';
          else if (isMedium) psFontName = 'EuclidCircularB-Medium';
          else psFontName = 'EuclidCircularB-Regular';
        } else if (/manrope/i.test(rawFont)) {
          canvasFont = 'Manrope';
          if (isBlack) psFontName = 'Manrope-ExtraBold';
          else if (isBold) psFontName = 'Manrope-Bold';
          else if (isSemiBold) psFontName = 'Manrope-SemiBold';
          else if (isMedium) psFontName = 'Manrope-Medium';
          else psFontName = 'Manrope-Regular';
        } else if (/sora/i.test(rawFont)) {
          canvasFont = 'Sora';
          if (isBold || isBlack) psFontName = 'Sora-Bold';
          else if (isSemiBold) psFontName = 'Sora-SemiBold';
          else if (isMedium) psFontName = 'Sora-Medium';
          else psFontName = 'Sora-Regular';
        } else if (/impact|bebas/i.test(rawFont)) {
          canvasFont = 'Impact';
          psFontName = 'Impact';
        } else if (/georgia|bodoni/i.test(rawFont)) {
          canvasFont = 'Georgia';
          psFontName = isBold ? 'Georgia-Bold' : 'Georgia';
        } else if (/times/i.test(rawFont)) {
          canvasFont = 'Times New Roman';
          psFontName = isBold ? 'TimesNewRomanPS-BoldMT' : 'TimesNewRomanPSMT';
        } else if (/nirmala|mangal/i.test(rawFont)) {
          canvasFont = 'Nirmala UI';
          psFontName = isBold ? 'NirmalaUI-Bold' : 'NirmalaUI';
        } else {
          // Default: POPPINS (universal modern font for English and Devanagari)
          canvasFont = 'Poppins';
          if (isItalic) psFontName = isBold ? 'Poppins-BoldItalic' : 'Poppins-Italic';
          else if (isBlack) psFontName = 'Poppins-Black';
          else if (isBold) psFontName = 'Poppins-Bold';
          else if (isSemiBold) psFontName = 'Poppins-SemiBold';
          else if (isMedium) psFontName = 'Poppins-Medium';
          else psFontName = 'Poppins-Regular';
        }
        
        // Calibrate font size and baseline so it fits typographical bounds without colliding
        const metrics = calibrateFontMetrics(rawFont, content, b.width, b.height, b.top);
        const fontSize = metrics.fontSize;
        const baselineY = metrics.baselineY;
        const align = (t.alignment || 'left').toLowerCase();
        const anchorX = align === 'center' ? b.left + b.width / 2 : (align === 'right' ? b.left + b.width : b.left);
        const leading = t.leading_pt || Math.round(fontSize * 1.18 * 10) / 10;
        const tracking = typeof t.tracking === 'number' ? t.tracking : 0;
        const angleDeg = typeof t.angle_deg === 'number' ? t.angle_deg : 0;
        let transform = [1, 0, 0, 1, anchorX, baselineY];
        if (Math.abs(angleDeg) > 1.5) {
          const rad = (angleDeg * Math.PI) / 180;
          const cos = Math.round(Math.cos(rad) * 10000) / 10000;
          const sin = Math.round(Math.sin(rad) * 10000) / 10000;
          transform = [cos, sin, -sin, cos, anchorX, baselineY];
        }

        // Render pixel canvas preview: generous padding guarantees ascenders, descenders, and Hindi matras are NEVER clipped
        const padX = Math.max(10, Math.round(fontSize * 0.45));
        const padY = Math.max(8, Math.round(fontSize * 0.40));
        const canvasLeft = Math.max(0, Math.round(b.left - padX));
        const canvasTop = Math.max(0, Math.round(b.top - padY));
        const canvasRight = Math.min(width, Math.round(b.left + b.width + padX));
        const canvasBottom = Math.min(height, Math.round(b.top + b.height + padY));
        const cw = Math.max(1, canvasRight - canvasLeft);
        const ch = Math.max(1, canvasBottom - canvasTop);

        const textCanvas = createCanvas(cw, ch);
        const tctx = textCanvas.getContext('2d');
        const weightStr = isBlack ? '900 ' : (isBold ? 'bold ' : (isSemiBold ? '600 ' : (isMedium ? '500 ' : '')));
        const styleStr = isItalic ? 'italic ' : '';
        tctx.font = `${styleStr}${weightStr}${fontSize}px "${canvasFont}"`;
        tctx.fillStyle = `rgb(${rgb.r}, ${rgb.g}, ${rgb.b})`;
        tctx.textAlign = align === 'center' ? 'center' : (align === 'right' ? 'right' : 'left');
        tctx.textBaseline = 'alphabetic';
        const drawX = align === 'center' ? Math.round(b.left + b.width / 2 - canvasLeft) : (align === 'right' ? Math.round(b.left + b.width - canvasLeft) : Math.round(b.left - canvasLeft));
        const drawY = Math.round(baselineY - canvasTop);
        tctx.fillText(content, drawX, drawY);

        return {
          name: info.name,
          canvas: textCanvas,
          top: canvasTop,
          left: canvasLeft,
          bottom: canvasBottom,
          right: canvasRight,
          visible: info.visible !== false,
          text: {
            text: content,
            transform: transform,
            style: {
              font: { name: psFontName },
              fontSize: fontSize,
              fillColor: { r: rgb.r, g: rgb.g, b: rgb.b },
              leading: leading,
              autoLeading: false,
              tracking: tracking
            },
            paragraphStyle: {
              justification: align === 'center' ? 'center' : (align === 'right' ? 'right' : 'left')
            }
          }
        };
      }
      // If dummy text, fall through to pixel cutout so we NEVER write literal "Text region 1"
    }

    // IF THIS IS A SHAPE / CTA CONTAINER LAYER: EMIT AS CRISP MATHEMATICAL VECTOR SHAPE!
    if (info.type === 'shape' && info.shape) {
      const shp = info.shape;
      const b = info.bounds || { left: 0, top: 0, width: 100, height: 40 };
      const bw = Math.max(1, b.width);
      const bh = Math.max(1, b.height);
      const canvas = createCanvas(bw, bh);
      const ctx = canvas.getContext('2d');

      const radius = shp.kind === 'pill' ? Math.round(bh / 2) : Math.max(0, Math.round(shp.corner_radius || 0));

      // Draw mathematical vector path
      ctx.beginPath();
      if (radius > 0) {
        const r = Math.min(radius, bw / 2, bh / 2);
        ctx.moveTo(r, 0);
        ctx.lineTo(bw - r, 0);
        ctx.quadraticCurveTo(bw, 0, bw, r);
        ctx.lineTo(bw, bh - r);
        ctx.quadraticCurveTo(bw, bh, bw - r, bh);
        ctx.lineTo(r, bh);
        ctx.quadraticCurveTo(0, bh, 0, bh - r);
        ctx.lineTo(0, r);
        ctx.quadraticCurveTo(0, 0, r, 0);
        ctx.closePath();
      } else {
        ctx.rect(0, 0, bw, bh);
      }

      // Fill with solid color or 2-stop linear gradient
      if (shp.gradient && Array.isArray(shp.gradient.stops) && shp.gradient.stops.length >= 2) {
        const angleDeg = shp.gradient.angle_deg || 0;
        let gx0 = 0, gy0 = 0, gx1 = 0, gy1 = 0;
        if (Math.abs(angleDeg - 90) < 45 || Math.abs(angleDeg - 270) < 45) {
          gx0 = 0; gy0 = 0;
          gx1 = 0; gy1 = bh;
        } else {
          gx0 = 0; gy0 = 0;
          gx1 = bw; gy1 = 0;
        }
        const grad = ctx.createLinearGradient(gx0, gy0, gx1, gy1);
        for (const stop of shp.gradient.stops) {
          grad.addColorStop(Math.max(0, Math.min(1, stop.pos)), stop.hex);
        }
        ctx.fillStyle = grad;
      } else {
        ctx.fillStyle = shp.fill_hex || '#FFFFFF';
      }
      ctx.fill();

      // Stroke border if present
      if (shp.stroke_hex && shp.stroke_width > 0) {
        ctx.strokeStyle = shp.stroke_hex;
        ctx.lineWidth = Math.max(1, shp.stroke_width * 2);
        ctx.stroke();
      }

      return {
        name: info.name,
        top: b.top,
        left: b.left,
        bottom: b.top + bh,
        right: b.left + bw,
        visible: info.visible !== false,
        opacity: (info.opacity == null ? 100 : info.opacity) / 100,
        canvas: canvas
      };
    }

    // Pixel layer (background, object, fallback)
    const assetFile = path.join(jobDir, `${info.id}.png`);
    if (fs.existsSync(assetFile)) {
      const img = await loadImage(assetFile);
      const canvas = createCanvas(img.width, img.height);
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);

      const b = info.bounds;
      return {
        name: info.name,
        top: b.top,
        left: b.left,
        bottom: b.top + img.height,
        right: b.left + img.width,
        visible: info.visible !== false,
        opacity: (info.opacity == null ? 100 : info.opacity) / 100,
        canvas: canvas
      };
    }

    return null;
  }

  const rootChildren = await convertNode(manifest.layer_tree);

  const psd = {
    width: width,
    height: height,
    children: rootChildren
  };

  const buffer = agPsd.writePsdBuffer(psd);
  const outPath = path.join(jobDir, 'decomposed.psd');
  fs.writeFileSync(outPath, buffer);
  console.log('ag-psd compiled PSD with native editable text layers:', outPath, 'size:', buffer.length);
}

main().catch(err => {
  console.error('ag-psd failed:', err);
  process.exit(1);
});

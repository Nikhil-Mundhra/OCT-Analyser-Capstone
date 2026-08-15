/**
 * SVG Vector Projection & Interactive Handle Drag Mechanics
 */

function buildSvgPathD(points) {
  if (!points || points.length === 0) return '';
  return 'M ' + points.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' L ');
}

function buildSfcmMaskPolygonD(topPoints, sfcmPoints) {
  if (!topPoints || topPoints.length === 0 || !sfcmPoints || sfcmPoints.length === 0) return '';
  const fwd = topPoints.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' L ');
  const rev = sfcmPoints.slice().reverse().map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' L ');
  return `M ${fwd} L ${rev} Z`;
}

function buildSvgHandlesHtml(points, layerType, indices = [8, 20, 32, 44, 56], isSfcm = false) {
  if (!points || points.length === 0) return '';
  let html = '';
  const handleClass = isSfcm ? 'sfcm-handle' : (layerType === 'top' ? 'top-handle' : 'bot-handle');
  indices.forEach(idx => {
    if (idx < points.length) {
      const p = points[idx];
      html += `<circle class="vector-handle ${handleClass}"
                       cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}"
                       r="3.5" data-layer="${layerType}" data-handle-y="${p[1].toFixed(1)}"
                       data-index="${idx}" />`;
    }
  });
  return html;
}

function onHandleDragStart(e, svgEl, layer, initialY) {
  e.preventDefault();
  e.stopPropagation();
  const circle = e.target;
  circle.classList.add('active');

  const pt = svgEl.createSVGPoint();
  pt.x = e.clientX;
  pt.y = e.clientY;
  const svgP = pt.matrixTransform(svgEl.getScreenCTM().inverse());

  draggingHandle = {
    circle: circle,
    svgEl: svgEl,
    layer: layer,
    startY: svgP.y,
    imgScale: parseFloat(svgEl.getAttribute('data-img-scale')) || 0.437,
    padT: parseFloat(svgEl.getAttribute('data-pad-t')) || 0,
    startMarginTop: parseFloat(document.getElementById('param-margin_top').value) || 15,
    startMarginBottom: parseFloat(document.getElementById('param-margin_bottom').value) || 15,
    startSfcmMargin: parseFloat(document.getElementById('param-sfcm_margin_bottom').value) || 15,
    startTopNoise: parseFloat(document.getElementById('param-top_noise_mult').value) || 1.5,
    startBotNoise: parseFloat(document.getElementById('param-bot_noise_mult').value) || 3.0,
    isSfcm: circle.classList.contains('sfcm-handle')
  };

  window.addEventListener('mousemove', onHandleDragMove);
  window.addEventListener('mouseup', onHandleDragEnd);
}

function onHandleDragMove(e) {
  if (!draggingHandle) return;
  const pt = draggingHandle.svgEl.createSVGPoint();
  pt.x = e.clientX;
  pt.y = e.clientY;
  const svgP = pt.matrixTransform(draggingHandle.svgEl.getScreenCTM().inverse());
  const deltaSvgY = svgP.y - draggingHandle.startY;

  if (draggingHandle.layer === 'top') {
    handleTopVectorDrag(deltaSvgY);
  } else {
    handleBottomVectorDrag(deltaSvgY);
  }
}

function handleTopVectorDrag(deltaSvgY) {
  const deltaRawY = -deltaSvgY / draggingHandle.imgScale;
  const deltaMargin = Math.round(deltaRawY);

  let newMargin = Math.max(5, Math.min(30, draggingHandle.startMarginTop + deltaMargin));
  const marginInput = document.getElementById('param-margin_top');
  marginInput.value = newMargin;
  updateSliderLabel('margin_top', newMargin + 'px');

  if (draggingHandle.startMarginTop + deltaMargin > 30) {
    const overflow = (draggingHandle.startMarginTop + deltaMargin) - 30;
    const multDelta = overflow * 0.08;
    let newTopMult = Math.max(0.5, Math.min(8.0, draggingHandle.startTopNoise + multDelta));
    newTopMult = Math.round(newTopMult * 10) / 10;
    const multInput = document.getElementById('param-top_noise_mult');
    multInput.value = newTopMult;
    updateSliderLabel('top_noise_mult', newTopMult.toFixed(1));
  } else if (draggingHandle.startMarginTop + deltaMargin < 5) {
    const underflow = 5 - (draggingHandle.startMarginTop + deltaMargin);
    const multDelta = underflow * 0.08;
    let newTopMult = Math.max(0.5, Math.min(8.0, draggingHandle.startTopNoise - multDelta));
    newTopMult = Math.round(newTopMult * 10) / 10;
    const multInput = document.getElementById('param-top_noise_mult');
    multInput.value = newTopMult;
    updateSliderLabel('top_noise_mult', newTopMult.toFixed(1));
  }

  scheduleDebouncedReprocess();
}

function handleBottomVectorDrag(deltaSvgY) {
  const deltaRawY = deltaSvgY / draggingHandle.imgScale;
  const deltaMargin = Math.round(deltaRawY);

  if (draggingHandle.isSfcm) {
    let newMargin = Math.max(0, Math.min(80, draggingHandle.startSfcmMargin + deltaMargin));
    const marginInput = document.getElementById('param-sfcm_margin_bottom');
    marginInput.value = newMargin;
    updateSliderLabel('sfcm_margin_bottom', newMargin + 'px');
  } else {
    let newMargin = Math.max(0, Math.min(80, draggingHandle.startMarginBottom + deltaMargin));
    const marginInput = document.getElementById('param-margin_bottom');
    marginInput.value = newMargin;
    updateSliderLabel('margin_bottom', newMargin + 'px');

    if (draggingHandle.startMarginBottom + deltaMargin > 80) {
      const overflow = (draggingHandle.startMarginBottom + deltaMargin) - 80;
      const multDelta = overflow * 0.05;
      let newBotMult = Math.max(1.5, Math.min(5.0, draggingHandle.startBotNoise + multDelta));
      newBotMult = Math.round(newBotMult * 10) / 10;
      const multInput = document.getElementById('param-bot_noise_mult');
      multInput.value = newBotMult;
      updateSliderLabel('bot_noise_mult', newBotMult.toFixed(1));
    }
  }

  scheduleDebouncedReprocess();
}

function onHandleDragEnd() {
  if (draggingHandle && draggingHandle.circle) {
    draggingHandle.circle.classList.remove('active');
  }
  draggingHandle = null;
  window.removeEventListener('mousemove', onHandleDragMove);
  window.removeEventListener('mouseup', onHandleDragEnd);
}

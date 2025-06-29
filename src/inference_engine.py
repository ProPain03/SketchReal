# Create: utils/smart_dual_inference.py
import cv2
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from torchvision import transforms
from typing import List, Dict, Any, Optional
from enum import Enum
import time
import os

class InputType(Enum):
    QUICKDRAW = "quickdraw"
    UPLOADED_IMAGE = "uploaded_image"
    AUTO_DETECT = "auto_detect"

class SmartDualInference:
    """
    Smart dual model system with adaptive fusion strategy:
    - For uploaded UI screenshots: Prioritize screenrecognition
    - For quickdraw sketches: Prioritize sketch detection + unified CNN
    """
    
    def __init__(self, unified_model_path: str, screenrec_model_path: str, config=None):
        self.config = config
        self.unified_model_path = unified_model_path
        self.screenrec_model_path = screenrec_model_path
        
        # Lazy loading - initialize as None
        self._unified_model = None
        self._screenrec_model = None
        self.unified_model_loaded = False
        self.screenrec_model_loaded = False
        
        print("🧠 Smart Dual System initialized (models will be loaded on demand)")
        
        # Initialize parameters for different modes
        self.sketch_params = {
            'min_area': 200,
            'max_area': 50000,
            'blur_kernel': 5,
            'threshold_block': 11,
            'threshold_c': 2,
            'padding': 15,
            'cluster_distance': 60
        }
        
        self.load_class_mappings()
    
    def load_class_mappings(self):
        """Load class mappings for both models"""
        self.unified_classes = self.config.UI_LABELS if self.config else {
            0: 'Other', 1: 'Background Image', 2: 'Checked View', 3: 'Icon',
            4: 'Input Field', 5: 'Image', 6: 'Text', 7: 'Text Button',
            8: 'Page Indicator', 9: 'Pop-Up Window', 10: 'Sliding Menu', 11: 'Switch'
        }
        
        self.screenrec_classes = {
            '0': 'Other', '1': 'Background', '2': 'Checked View', '3': 'Icon',
            '4': 'Input Field', '5': 'Image', '6': 'Text', '7': 'Text Button',
            '8': 'Page Indicator', '9': 'Pop-Up Window', '10': 'Sliding Menu', '11': 'Switch'
        }

    def detect_input_type(self, image: np.ndarray) -> InputType:
        """Automatically detect if image is a quickdraw sketch or UI screenshot"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate image properties
        height, width = gray.shape
        total_pixels = height * width
        
        # 1. Check for mostly white/empty background (quickdraw characteristic)
        white_threshold = 240
        white_pixels = np.sum(gray > white_threshold)
        white_ratio = white_pixels / total_pixels
        
        # 2. Check for sparse content (quickdraw characteristic)
        content_threshold = 200
        content_pixels = np.sum(gray < content_threshold)
        content_ratio = content_pixels / total_pixels
        
        # 3. Check for simple shapes (quickdraw characteristic)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        edge_ratio = edge_pixels / total_pixels
        
        # 4. Check for color diversity (UI screenshots usually have more colors)
        unique_colors = len(np.unique(gray))
        color_diversity = unique_colors / 256.0
        
        # Decision logic
        quickdraw_score = 0
        
        # High white background suggests quickdraw
        if white_ratio > 0.7:
            quickdraw_score += 2
        
        # Low content ratio suggests quickdraw
        if content_ratio < 0.3:
            quickdraw_score += 2
        
        # Simple edge structure suggests quickdraw
        if 0.01 < edge_ratio < 0.1:
            quickdraw_score += 1
        
        # Low color diversity suggests quickdraw
        if color_diversity < 0.3:
            quickdraw_score += 1
        
        # Decision threshold
        if quickdraw_score >= 3:
            return InputType.QUICKDRAW
        else:
            return InputType.UPLOADED_IMAGE
    
    def detect_sketch_elements(self, image: np.ndarray, confidence_threshold: float = 0.4) -> List[Dict]:
        """Enhanced sketch detection for quickdraw mode"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # More sensitive thresholding for pencil-like strokes
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 4
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Adjusted area thresholds for quickdraw
            if 150 < area < 30000:
                x, y, w, h = cv2.boundingRect(contour)
                
                padding = 10
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image.shape[1], x + w + padding)
                y2 = min(image.shape[0], y + h + padding)
                
                if (x2 - x1) < 20 or (y2 - y1) < 20:
                    continue
                
                # Enhanced confidence calculation for quickdraw
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    compactness = 4 * np.pi * area / (perimeter * perimeter)
                    aspect_ratio = w / h if h > 0 else 1
                    
                    stroke_confidence = min(0.9, max(0.4, compactness * 0.8))
                    size_confidence = min(1.0, area / 5000)
                    
                    confidence = (stroke_confidence + size_confidence) / 2
                else:
                    confidence = 0.5
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'contour_id': i,
                    'source': 'sketch_detector'
                })
        
        # Sort by confidence and area
        detections.sort(key=lambda x: (x['confidence'], x['area']), reverse=True)
        
        return detections
    
    def classify_sketch_crop(self, image: np.ndarray, bbox: List[int]) -> Dict:
        """Classify sketch crop using unified model"""
        try:
            if self._unified_model is None:
                self._load_unified_model()
            
            x1, y1, x2, y2 = bbox
            crop = image[y1:y2, x1:x2]
            
            if crop.size == 0:
                return {'class_id': 0, 'class_name': 'Other', 'confidence': 0.0}
            
            # Resize and preprocess
            crop_resized = cv2.resize(crop, (64, 64))
            crop_normalized = crop_resized.astype(np.float32) / 255.0
            crop_batch = np.expand_dims(crop_normalized, axis=0)
            
            # Use the model
            predictions = self._unified_model.predict(crop_batch, verbose=0)
            
            class_id = np.argmax(predictions[0])
            confidence = float(predictions[0][class_id])
            class_name = self.unified_classes.get(class_id, f'Unknown_{class_id}')
            
            return {
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'source': 'unified_cnn'
            }
        except Exception as e:
            print(f"⚠️ Error in sketch classification: {e}")
            return {'class_id': 0, 'class_name': 'Other', 'confidence': 0.0}
    
    def detect_screenrecognition(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """Screenrecognition detection using lazy-loaded model"""
        try:
            if self._screenrec_model is None:
                self._load_screenrec_model()
            
            if self._screenrec_model is None:
                print("⚠️ Screenrecognition model not available")
                return []
            
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img_transforms = transforms.ToTensor()
            img_tensor = img_transforms(pil_image)
            
            with torch.no_grad():
                predictions = self._screenrec_model([img_tensor])
            
            detections = []
            pred_data = predictions[1][0]
            
            for i in range(len(pred_data['boxes'])):
                confidence = float(pred_data['scores'][i])
                
                if confidence > confidence_threshold:
                    x1, y1, x2, y2 = pred_data['boxes'][i]
                    class_id = int(pred_data['labels'][i])
                    class_name = self.screenrec_classes.get(str(class_id), f'Unknown_{class_id}')
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'source': 'screenrecognition'
                    })
            
            return detections
        except Exception as e:
            print(f"⚠️ Screenrec detection error: {e}")
            return []

    # ============== ENHANCED FILTERING METHODS ==============
    
    def filter_unified_detections_with_screenrec(self, unified_detections: List[Dict], 
                                               screenrec_detections: List[Dict], 
                                               overlap_threshold: float = 0.6,
                                               confidence_boost: float = 0.1) -> List[Dict]:
        """Filter unified model detections using screenrecognition as ground truth"""
        if not screenrec_detections:
            return self.apply_basic_unified_filtering(unified_detections)
        
        filtered_detections = []
        suppressed_count = 0
        boosted_count = 0
        
        print(f"   🔍 Filtering {len(unified_detections)} unified detections using {len(screenrec_detections)} screenrec detections...")
        
        for unified_det in unified_detections:
            should_keep = True
            max_overlap = 0.0
            matching_screenrec = None
            
            # Check overlap with all screenrec detections
            for screenrec_det in screenrec_detections:
                overlap = self.compute_iou(unified_det['bbox'], screenrec_det['bbox'])
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    matching_screenrec = screenrec_det
                
                # Suppression logic
                if overlap > overlap_threshold:
                    # Case 1: Same class and high overlap - likely duplicate
                    if self.are_similar_classes(unified_det.get('class_name', ''), 
                                              screenrec_det.get('class_name', '')):
                        should_keep = False
                        suppressed_count += 1
                        break
                    
                    # Case 2: Different classes but unified detection is likely noise
                    elif self.is_likely_noise(unified_det, screenrec_det):
                        should_keep = False
                        suppressed_count += 1
                        break
            
            if should_keep:
                # Apply confidence boost if there's good alignment
                if max_overlap > 0.3 and matching_screenrec:
                    unified_det['confidence'] = min(0.95, 
                        unified_det.get('confidence', 0.5) + confidence_boost)
                    unified_det['alignment_score'] = max_overlap
                    unified_det['aligned_with'] = matching_screenrec.get('class_name', 'unknown')
                    boosted_count += 1
                
                filtered_detections.append(unified_det)
        
        print(f"   📊 Filtering results:")
        print(f"      • Original unified detections: {len(unified_detections)}")
        print(f"      • Suppressed as redundant: {suppressed_count}")
        print(f"      • Confidence boosted: {boosted_count}")
        print(f"      • Final unified detections: {len(filtered_detections)}")
        
        return filtered_detections

    def are_similar_classes(self, class1: str, class2: str) -> bool:
        """Check if two class names represent similar UI elements"""
        similarity_groups = [
            {'Text', 'Label', 'Text Button', 'Link'},
            {'Image', 'Background Image', 'Background', 'Icon'},
            {'Input Field', 'Search Box', 'Text Input'},
            {'Button', 'Text Button', 'Icon Button'},
            {'Other', 'Unknown', 'Misc'}
        ]
        
        # Normalize class names
        class1_norm = class1.lower().strip()
        class2_norm = class2.lower().strip()
        
        # Check if they're in the same similarity group
        for group in similarity_groups:
            group_lower = {cls.lower() for cls in group}
            if class1_norm in group_lower and class2_norm in group_lower:
                return True
        
        # Check for partial matches
        if 'text' in class1_norm and 'text' in class2_norm:
            return True
        if 'image' in class1_norm and 'image' in class2_norm:
            return True
        if 'button' in class1_norm and 'button' in class2_norm:
            return True
        
        return False

    def is_likely_noise(self, unified_det: Dict, screenrec_det: Dict) -> bool:
        """Determine if unified detection is likely noise given a conflicting screenrec detection"""
        unified_class = unified_det.get('class_name', '').lower()
        unified_conf = unified_det.get('confidence', 0.0)
        unified_bbox = unified_det['bbox']
        
        screenrec_class = screenrec_det.get('class_name', '').lower()
        screenrec_conf = screenrec_det.get('confidence', 0.0)
        screenrec_bbox = screenrec_det['bbox']
        
        # Calculate areas
        unified_area = (unified_bbox[2] - unified_bbox[0]) * (unified_bbox[3] - unified_bbox[1])
        screenrec_area = (screenrec_bbox[2] - screenrec_bbox[0]) * (screenrec_bbox[3] - screenrec_bbox[1])
        
        # Case 1: Unified detection is much smaller and low confidence
        if unified_area < screenrec_area * 0.3 and unified_conf < 0.6:
            return True
        
        # Case 2: Unified detects generic classes while screenrec detects specific UI
        generic_classes = {'image', 'background', 'background image', 'other', 'unknown'}
        specific_ui_classes = {'text', 'button', 'input field', 'icon', 'label', 'link'}
        
        if (unified_class in generic_classes and 
            any(ui_class in screenrec_class for ui_class in specific_ui_classes) and
            screenrec_conf > 0.7):
            return True
        
        # Case 3: Very low confidence unified detection
        if unified_conf < 0.4 and screenrec_conf > 0.6:
            return True
        
        # Case 4: Character fragment detection
        if (self.is_likely_character_fragment(unified_det) and 
            'text' in screenrec_class and screenrec_conf > 0.6):
            return True
        
        return False

    def is_likely_character_fragment(self, detection: Dict) -> bool:
        """Check if detection is likely a single character or small fragment"""
        bbox = detection['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        
        # Very small detections are likely character fragments
        if area < 500:
            return True
        
        # Very thin or very short detections
        aspect_ratio = max(width, height) / min(width, height)
        if (width < 25 or height < 15) and aspect_ratio < 3:
            return True
        
        # Low confidence small detections of generic classes
        conf = detection.get('confidence', 0.0)
        class_name = detection.get('class_name', '').lower()
        generic_classes = {'image', 'background', 'other', 'unknown'}
        
        if (area < 1000 and conf < 0.7 and 
            any(generic in class_name for generic in generic_classes)):
            return True
        
        return False

    def apply_basic_unified_filtering(self, unified_detections: List[Dict]) -> List[Dict]:
        """Apply basic filtering when no screenrec detections are available"""
        filtered_detections = []
        
        for detection in unified_detections:
            # Filter out very small, likely noise detections
            if not self.is_likely_character_fragment(detection):
                # Apply minimum confidence threshold
                if detection.get('confidence', 0.0) > 0.5:
                    filtered_detections.append(detection)
        
        return filtered_detections

    def enhanced_sketch_detection_filtering(self, sketch_detections: List[Dict], 
                                          screenrec_detections: List[Dict]) -> List[Dict]:
        """Enhanced filtering for sketch detections to reduce false positives"""
        if not sketch_detections:
            return []
        
        # First apply the screenrec-based filtering
        filtered_sketches = self.filter_unified_detections_with_screenrec(
            sketch_detections, screenrec_detections, overlap_threshold=0.5
        )
        
        # Additional sketch-specific filtering
        final_sketches = []
        
        for sketch in filtered_sketches:
            # Skip very small detections (likely character fragments)
            if self.is_likely_character_fragment(sketch):
                continue
            
            # Apply stricter confidence threshold for sketches
            if sketch.get('confidence', 0.0) < 0.6:
                continue
            
            # Check for reasonable aspect ratios
            bbox = sketch['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            aspect_ratio = max(width, height) / min(width, height)
            
            # Skip extremely elongated detections (likely artifacts)
            if aspect_ratio > 10:
                continue
            
            final_sketches.append(sketch)
        
        return final_sketches

    def cluster_and_merge_fragments(self, detections: List[Dict], 
                                  distance_threshold: float = 30) -> List[Dict]:
        """Cluster nearby small detections that might be character fragments and merge them into larger text blocks"""
        if len(detections) <= 1:
            return detections
        
        # Separate small detections (potential fragments) from larger ones
        fragments = []
        complete_detections = []
        
        for det in detections:
            bbox = det['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            if area < 800 and det.get('confidence', 0.0) < 0.8:  # Potential fragment
                fragments.append(det)
            else:
                complete_detections.append(det)
        
        if len(fragments) < 2:
            return detections  # Not enough fragments to cluster
        
        # Cluster fragments by proximity
        clusters = []
        used_fragments = set()
        
        for i, frag1 in enumerate(fragments):
            if i in used_fragments:
                continue
            
            cluster = [frag1]
            used_fragments.add(i)
            
            # Find nearby fragments
            for j, frag2 in enumerate(fragments[i+1:], i+1):
                if j in used_fragments:
                    continue
                
                # Check if fragments are close enough
                dist = self.bbox_distance(frag1['bbox'], frag2['bbox'])
                if dist < distance_threshold:
                    cluster.append(frag2)
                    used_fragments.add(j)
            
            if len(cluster) >= 2:  # Only merge if we have multiple fragments
                clusters.append(cluster)
            else:
                complete_detections.extend(cluster)  # Keep as individual detection
        
        # Merge clustered fragments
        merged_detections = []
        for cluster in clusters:
            merged_bbox = self._merge_boxes([det['bbox'] for det in cluster])
            avg_confidence = sum(det.get('confidence', 0.0) for det in cluster) / len(cluster)
            
            # Use the most common class name in the cluster
            class_names = [det.get('class_name', 'Text') for det in cluster]
            most_common_class = max(set(class_names), key=class_names.count)
            
            merged_detection = {
                'bbox': merged_bbox,
                'confidence': avg_confidence + 0.1,  # Slight boost for merged detection
                'class_name': most_common_class,
                'source': 'merged_fragments',
                'fragment_count': len(cluster)
            }
            merged_detections.append(merged_detection)
        
        return complete_detections + merged_detections

    def bbox_distance(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate minimum distance between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate center points
        cx1, cy1 = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
        cx2, cy2 = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
        
        # Euclidean distance between centers
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

    # ============== FUSION METHODS ==============
    
    def smart_fusion(self, sketch_dets: List[Dict], screenrec_dets: List[Dict], 
                input_type: InputType, iou_threshold: float = 0.3, 
                strict_ui_mode: bool = True) -> List[Dict]:
        """Smart fusion strategy with optional strict mode"""
        
        if input_type == InputType.QUICKDRAW:
            return self.fusion_quickdraw_priority(sketch_dets, screenrec_dets, iou_threshold)
        else:  # UPLOADED_IMAGE
            if strict_ui_mode:
                return self.fusion_screenrec_priority(sketch_dets, screenrec_dets, iou_threshold)
            else:
                return self.fusion_screenrec_only(sketch_dets, screenrec_dets, iou_threshold)
    
    def fusion_quickdraw_priority(self, sketch_dets: List[Dict], screenrec_dets: List[Dict], 
                                  iou_threshold: float) -> List[Dict]:
        """Fusion with sketch detection priority (for quickdraw)"""
        fused_results = []
        
        # Add all sketch detections first (high priority)
        for sketch_det in sketch_dets:
            sketch_det['fusion_type'] = 'sketch_primary'
            sketch_det['priority'] = 'high'
            fused_results.append(sketch_det)
        
        # Add non-overlapping screenrec detections
        for screenrec_det in screenrec_dets:
            is_overlapping = False
            
            for sketch_det in sketch_dets:
                iou = self.compute_iou(sketch_det['bbox'], screenrec_det['bbox'])
                if iou > iou_threshold:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                screenrec_det['fusion_type'] = 'screenrec_supplementary'
                screenrec_det['priority'] = 'medium'
                fused_results.append(screenrec_det)
        
        return fused_results
    
    def fusion_screenrec_priority(self, sketch_dets: List[Dict], screenrec_dets: List[Dict], 
                              iou_threshold: float) -> List[Dict]:
        """ULTRA-STRICT fusion with screenrecognition absolute priority for UI screenshots"""
        fused_results = []
        
        # Step 1: Add ALL screenrec detections with highest priority
        for screenrec_det in screenrec_dets:
            screenrec_det['fusion_type'] = 'screenrec_primary'
            screenrec_det['priority'] = 'ultra_high'
            fused_results.append(screenrec_det)
        
        # Step 2: VERY STRICT filtering for sketch detections
        high_quality_sketch_dets = []
        
        for sketch_det in sketch_dets:
            # Check 1: Zero overlap with screenrec detections
            has_any_overlap = False
            
            for screenrec_det in screenrec_dets:
                iou = self.compute_iou(sketch_det['bbox'], screenrec_det['bbox'])
                if iou > 0.1:  # Even tiny overlap = reject
                    has_any_overlap = True
                    break
            
            if has_any_overlap:
                continue  # Skip this sketch detection
            
            # Check 2: High confidence threshold
            sketch_confidence = sketch_det.get('confidence', 0)
            if sketch_confidence < 0.7:  # Very high threshold
                continue
            
            # Check 3: Reasonable size (not tiny fragments)
            x1, y1, x2, y2 = sketch_det['bbox']
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Reject very small detections (likely noise/fragments)
            if width < 40 or height < 40 or area < 2000:
                continue
            
            # Check 4: Aspect ratio (reject very thin/elongated detections)
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 8:  # Very elongated = likely fragment
                continue
            
            # If it passes all checks, it's high quality
            high_quality_sketch_dets.append(sketch_det)
        
        # Step 3: Add only the BEST sketch detections (limit quantity)
        high_quality_sketch_dets.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Take only top 3 high-quality sketch detections (or fewer)
        max_supplementary = min(3, len(high_quality_sketch_dets))
        
        for sketch_det in high_quality_sketch_dets[:max_supplementary]:
            sketch_det['fusion_type'] = 'sketch_supplementary_strict'
            sketch_det['priority'] = 'low'
            fused_results.append(sketch_det)
        
        print(f"   📊 Fusion details:")
        print(f"      • Screenrec detections: {len(screenrec_dets)} (all kept)")
        print(f"      • Sketch candidates: {len(sketch_dets)}")
        print(f"      • High-quality sketches: {len(high_quality_sketch_dets)}")
        print(f"      • Final sketch additions: {max_supplementary}")
        
        return fused_results
    
    def fusion_screenrec_only(self, sketch_dets: List[Dict], screenrec_dets: List[Dict], 
                          iou_threshold: float) -> List[Dict]:
        """Pure screenrecognition mode - NO sketch detections added"""
        fused_results = []
        
        # Add ONLY screenrec detections
        for screenrec_det in screenrec_dets:
            screenrec_det['fusion_type'] = 'screenrec_only'
            screenrec_det['priority'] = 'absolute'
            fused_results.append(screenrec_det)
        
        print(f"   📊 Pure Screenrec Mode:")
        print(f"      • Screenrec detections: {len(screenrec_dets)} (all kept)")
        print(f"      • Sketch detections: {len(sketch_dets)} (all ignored)")
        
        return fused_results
        
    def adaptive_confidence_threshold(self, image_path, base_threshold=0.5):
        """Adjust confidence threshold based on image complexity"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return base_threshold
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 100, 200)
            
            # Count edge pixels as a measure of complexity
            edge_pixel_count = np.count_nonzero(edges)
            total_pixels = edges.shape[0] * edges.shape[1]
            complexity = edge_pixel_count / total_pixels
            
            # Adjust threshold based on complexity
            if complexity > 0.1:  # High complexity
                return min(base_threshold + 0.2, 0.8)  # Increase threshold
            elif complexity < 0.03:  # Low complexity
                return max(base_threshold - 0.1, 0.3)  # Decrease threshold
            else:
                return base_threshold  # Use default
        except Exception:
            return base_threshold  # Fall back to default if anything fails

    def inference_pipeline(self, image_path: str, input_type: Optional[InputType] = None,
                      sketch_conf: float = 0.4, screenrec_conf: float = 0.5, 
                      iou_threshold: float = 0.3, strict_ui_mode: bool = True, 
                      use_quantization: bool = False, 
                      enable_fragment_filtering: bool = True) -> Dict[str, Any]:
        """Enhanced inference pipeline with fragment filtering"""
        print(f"🧠 Enhanced Smart Processing: {image_path}")
        print(f"Parameters: input_type={input_type}, sketch_conf={sketch_conf}, screenrec_conf={screenrec_conf}")
        print(f"Advanced filtering: fragment_filtering={enable_fragment_filtering}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Image shape: {image.shape}")
        
        # Auto-detect input type if not specified
        if input_type is None:
            input_type = self.detect_input_type(image)
        
        print(f"🎯 Detected input type: {input_type.value}")

        # Load appropriate models
        if input_type == InputType.QUICKDRAW:
            if self._unified_model is None:
                self._load_unified_model(use_quantization=use_quantization)
        else:
            if self._screenrec_model is None:
                self._load_screenrec_model(use_quantization=use_quantization)
        
        # Step 1: Screenrecognition Detection (run first to use as filter)
        print("🖥️ Step 1: Screenrecognition detection...")
        adaptive_threshold = self.adaptive_confidence_threshold(image_path, base_threshold=screenrec_conf)
        screenrec_detections = self.detect_screenrecognition(image, adaptive_threshold)
        print(f"   Found {len(screenrec_detections)} UI elements")
        
        # Apply advanced NMS and text merging to screenrec results
        if len(screenrec_detections) > 30:
            print("   Applying advanced NMS...")
            boxes = [det['bbox'] for det in screenrec_detections]
            scores = [det['confidence'] for det in screenrec_detections]
            classes = [det['class_name'] for det in screenrec_detections]
            
            keep_indices = self.advanced_nms(boxes, scores, classes, iou_threshold=0.5)
            screenrec_detections = [screenrec_detections[i] for i in keep_indices]
            print(f"   Reduced to {len(screenrec_detections)} detections")
        
        print("   Merging text elements...")
        screenrec_detections = self.merge_text_elements(screenrec_detections, distance_threshold=15)
        print(f"   Final screenrec detections: {len(screenrec_detections)}")
        
        # Step 2: Sketch Detection + Classification with Enhanced Filtering
        print("🎨 Step 2: Enhanced sketch detection + filtering...")
        raw_sketch_detections = self.detect_sketch_elements(image, sketch_conf)
        print(f"   Raw sketch detections: {len(raw_sketch_detections)}")
        
        # Classify raw detections
        unified_detections = []
        for detection in raw_sketch_detections:
            classification = self.classify_sketch_crop(image, detection['bbox'])
            unified_detection = {**detection, **classification}
            unified_detections.append(unified_detection)
        
        # Apply enhanced filtering using screenrec as ground truth
        if enable_fragment_filtering:
            print("   🔍 Applying enhanced filtering...")
            
            # Filter out redundant/noise detections using screenrec
            filtered_unified = self.filter_unified_detections_with_screenrec(
                unified_detections, screenrec_detections, overlap_threshold=0.6
            )
            
            # Apply additional sketch-specific filtering
            final_unified = self.enhanced_sketch_detection_filtering(
                filtered_unified, screenrec_detections
            )
            
            # Try to merge character fragments into text blocks
            final_unified = self.cluster_and_merge_fragments(final_unified, distance_threshold=40)
            
            unified_detections = final_unified
            print(f"   Final unified detections after filtering: {len(unified_detections)}")
        
        # Step 3: Smart Fusion
        print(f"🧠 Step 3: Smart fusion ({input_type.value} mode, strict={strict_ui_mode})...")
        fused_results = self.smart_fusion(
            unified_detections, screenrec_detections, input_type, iou_threshold, strict_ui_mode
        )
        print(f"   Final results: {len(fused_results)} objects")
        
        return {
            'image_path': image_path,
            'image_shape': image.shape,
            'input_type': input_type.value,
            'sketch_detections': unified_detections,
            'screenrec_detections': screenrec_detections,
            'fused_results': fused_results,
            'filtering_stats': {
                'raw_sketch_count': len(raw_sketch_detections),
                'filtered_sketch_count': len(unified_detections),
                'screenrec_count': len(screenrec_detections),
                'final_fused_count': len(fused_results)
            },
            'fusion_stats': {
                'input_type': input_type.value,
                'total_sketches': len(unified_detections),
                'total_screenrec': len(screenrec_detections),
                'total_fused': len(fused_results),
                'fusion_types': self.get_fusion_breakdown(fused_results)
            }
        }

    # ============== UTILITY METHODS ==============

    def compute_iou(self, box1: List[int], box2: List[int]) -> float:
        """Compute IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def get_fusion_breakdown(self, fused_results: List[Dict]) -> Dict[str, int]:
        """Get breakdown of fusion types"""
        breakdown = {}
        for result in fused_results:
            fusion_type = result.get('fusion_type', 'unknown')
            breakdown[fusion_type] = breakdown.get(fusion_type, 0) + 1
        return breakdown

    def advanced_nms(self, boxes, scores, classes, iou_threshold=0.5, class_specific=True, hierarchical=True):
        """Advanced Non-Maximum Suppression for UI elements"""
        if len(boxes) == 0:
            return []
        
        # Convert to numpy arrays if not already
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        # Get areas of all boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by confidence score
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Get overlapping boxes with the current highest score box
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            # Compute intersection area
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            # Compute IoU
            overlap = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            # Special handling for hierarchical UI elements
            if hierarchical:
                # Check if current box contains other boxes (parent-child relationship)
                is_contained = (x1[order[1:]] >= x1[i]) & (y1[order[1:]] >= y1[i]) & \
                            (x2[order[1:]] <= x2[i]) & (y2[order[1:]] <= y2[i])
                
                # Adjust overlap scores for contained elements
                contained_overlap = np.where(
                    is_contained & (overlap < 0.9) & (classes[order[1:]] != classes[i]),
                    0.0,  # Don't suppress different classes contained within
                    overlap
                )
                
                # But if they're the same class and highly overlapping, likely duplicates
                overlap = np.where(
                    (classes[order[1:]] == classes[i]) & (overlap > 0.7),
                    1.0,  # Force suppression for same-class high overlap
                    contained_overlap
                )
            elif class_specific:
                # For class-specific NMS, only suppress boxes of the same class
                overlap = np.where(
                    classes[order[1:]] == classes[i],
                    overlap,
                    0.0
                )
            
            # Get indices of boxes to keep
            inds = np.where(overlap <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep

    def merge_text_elements(self, detections, distance_threshold=10):
        """Merge nearby text elements that are likely part of the same text block"""
        if len(detections) <= 1:
            return detections
        
        # First, separate text elements from other elements
        text_elements = []
        other_elements = []
        
        text_classes = ['Text', 'Label', 'Link', 'Title']
        
        for det in detections:
            if any(text_class in det.get('class_name', '') for text_class in text_classes):
                text_elements.append(det)
            else:
                other_elements.append(det)
        
        # If we have few text elements, no need to merge
        if len(text_elements) <= 1:
            return detections
        
        # Sort text elements by y-coordinate (top to bottom)
        text_elements.sort(key=lambda x: x['bbox'][1])
        
        # Merge text elements in the same line
        merged_text = []
        current_group = [text_elements[0]]
        
        for i in range(1, len(text_elements)):
            current = text_elements[i]
            prev = current_group[-1]
            
            # Check if elements are on the same line (similar y-coordinate)
            y_diff = abs(current['bbox'][1] - prev['bbox'][1])
            height = max(prev['bbox'][3] - prev['bbox'][1], current['bbox'][3] - current['bbox'][1])
            
            if y_diff < height * 0.5:  # Same line if vertical difference is less than half the height
                # Check horizontal distance
                if current['bbox'][0] - prev['bbox'][2] < distance_threshold:
                    # Close enough horizontally, add to current group
                    current_group.append(current)
                else:
                    # Too far horizontally, create new group
                    merged_box = self._merge_boxes([det['bbox'] for det in current_group])
                    avg_conf = sum(det['confidence'] for det in current_group) / len(current_group)
                    
                    merged_text.append({
                        'class_name': 'Text Block',
                        'confidence': avg_conf,
                        'bbox': merged_box,
                        'fusion_type': current_group[0]['fusion_type'] if 'fusion_type' in current_group[0] else 'merged_text'
                    })
                    current_group = [current]
            else:
                # New line, create new group
                merged_box = self._merge_boxes([det['bbox'] for det in current_group])
                avg_conf = sum(det['confidence'] for det in current_group) / len(current_group)
                
                merged_text.append({
                    'class_name': 'Text Block',
                    'confidence': avg_conf,
                    'bbox': merged_box,
                    'fusion_type': current_group[0]['fusion_type'] if 'fusion_type' in current_group[0] else 'merged_text'
                })
                current_group = [current]
        
        # Add the last group
        if current_group:
            merged_box = self._merge_boxes([det['bbox'] for det in current_group])
            avg_conf = sum(det['confidence'] for det in current_group) / len(current_group)
            
            merged_text.append({
                'class_name': 'Text Block',
                'confidence': avg_conf,
                'bbox': merged_box,
                'fusion_type': current_group[0]['fusion_type'] if 'fusion_type' in current_group[0] else 'merged_text'
            })
        
        # Combine merged text with other elements
        return other_elements + merged_text

    def _merge_boxes(self, boxes):
        """Merge multiple bounding boxes into one enclosing box"""
        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)
        return [x1, y1, x2, y2]

    def _load_unified_model(self, use_quantization=False):
        """Load the unified CNN model with optional quantization"""
        try:
            self._unified_model = tf.keras.models.load_model(self.unified_model_path)
            print(f"✅ Unified CNN loaded: {self.unified_model_path}")
            self.unified_model_loaded = True
        except Exception as e:
            print(f"❌ Failed to load unified model: {e}")
            self._unified_model = None

    def _load_screenrec_model(self, use_quantization=False):
        """Load the screenrecognition model with optional quantization"""
        try:
            self._screenrec_model = torch.jit.load(self.screenrec_model_path, map_location='cpu')
            print(f"✅ Screenrecognition model loaded: {self.screenrec_model_path}")
            self.screenrec_model_loaded = True
        except Exception as e:
            print(f"❌ Failed to load screenrecognition model: {e}")
            self._screenrec_model = None



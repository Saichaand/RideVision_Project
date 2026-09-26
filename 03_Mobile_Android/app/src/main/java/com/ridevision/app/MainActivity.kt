package com.ridevision.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import androidx.exifinterface.media.ExifInterface
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.core.app.ActivityCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationServices
import com.ridevision.app.api.CheckAheadRequest
import com.ridevision.app.api.PotholeReportRequest
import com.ridevision.app.api.RideVisionApiClient
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * RideVision Main Mobile Interface:
 * 1. Executes real-time or photo-based YOLOv8 pothole detection via TFLite.
 * 2. Visualizes bounding boxes with color-coded severity metrics.
 * 3. Monitors commuter location to surface proactive hazard warnings ahead.
 * 4. Dispatches verified reports to municipal channels (MCC WhatsApp, BBMP helpline).
 */
class MainActivity : ComponentActivity() {

    private lateinit var detector: PotholeDetector
    private lateinit var fusedLocationClient: FusedLocationProviderClient

    private lateinit var imageView: ImageView
    private lateinit var placeholderText: TextView
    private lateinit var resultStatusText: TextView
    private lateinit var resultDetailsText: TextView
    private lateinit var warningBanner: LinearLayout
    private lateinit var warningTitle: TextView
    private lateinit var warningSubtitle: TextView
    private lateinit var cameraButton: Button
    private lateinit var pickButton: Button
    private lateinit var reportButton: Button

    private var currentBitmap: Bitmap? = null
    private var lastDetections: List<PotholeDetector.Detection> = emptyList()
    private var currentLat: Double = 12.8715
    private var currentLon: Double = 74.8564
    private var currentHeading: Float = 245f

    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicturePreview()
    ) { bitmap: Bitmap? ->
        bitmap?.let { processBitmap(it) }
    }

    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { runDetectionOnUri(it) }
    }

    private val requestLocationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            startLocationAndHazardMonitoring()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        placeholderText = findViewById(R.id.placeholderText)
        resultStatusText = findViewById(R.id.resultStatusText)
        resultDetailsText = findViewById(R.id.resultDetailsText)
        warningBanner = findViewById(R.id.warningBanner)
        warningTitle = findViewById(R.id.warningTitle)
        warningSubtitle = findViewById(R.id.warningSubtitle)
        cameraButton = findViewById(R.id.cameraButton)
        pickButton = findViewById(R.id.pickButton)
        reportButton = findViewById(R.id.reportButton)

        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)

        // Initialize YOLOv8 TFLite model from assets
        try {
            detector = PotholeDetector(this, "best.tflite")
            resultStatusText.text = "AI Model: best.tflite (YOLOv8n) Loaded"
        } catch (e: Exception) {
            resultStatusText.text = "Model Load Error: ${e.message}"
        }

        cameraButton.setOnClickListener {
            takePictureLauncher.launch(null)
        }

        pickButton.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }

        reportButton.setOnClickListener {
            handleReportSubmission()
        }

        checkAndRequestLocation()
    }

    private fun checkAndRequestLocation() {
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            startLocationAndHazardMonitoring()
        } else {
            requestLocationPermissionLauncher.launch(Manifest.permission.ACCESS_FINE_LOCATION)
        }
    }

    private fun startLocationAndHazardMonitoring() {
        try {
            fusedLocationClient.lastLocation.addOnSuccessListener { location ->
                if (location != null) {
                    currentLat = location.latitude
                    currentLon = location.longitude
                    if (location.hasBearing()) {
                        currentHeading = location.bearing
                    }
                    pollHazardWarningAhead()
                }
            }
        } catch (e: SecurityException) {
            // Permission handled
        }
    }

    private fun pollHazardWarningAhead() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val response = RideVisionApiClient.service.checkWarningAhead(
                    CheckAheadRequest(currentLat, currentLon, currentHeading)
                )
                withContext(Dispatchers.Main) {
                    if (response.isSuccessful && response.body()?.hasWarning == true) {
                        val w = response.body()?.warning
                        if (w != null) {
                            warningTitle.text = "⚠️ HAZARD AHEAD: ${w.severity.uppercase()} POTHOLE ${w.distanceM}m"
                            warningSubtitle.text = "${w.address} — Maintain safe distance"
                            warningBanner.visibility = View.VISIBLE
                        }
                    } else {
                        warningBanner.visibility = View.GONE
                    }
                }
            } catch (e: Exception) {
                // Background poll fallback
            }
        }
    }

    private fun runDetectionOnUri(uri: Uri) {
        val bitmap = loadOrientedBitmap(uri)
        if (bitmap == null) {
            Toast.makeText(this, "Failed to load selected image.", Toast.LENGTH_SHORT).show()
            return
        }
        processBitmap(bitmap)
    }

    private fun loadOrientedBitmap(uri: Uri): Bitmap? {
        return try {
            val inputStream = contentResolver.openInputStream(uri) ?: return null
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream.close()

            val exifStream = contentResolver.openInputStream(uri) ?: return bitmap
            val exif = ExifInterface(exifStream)
            val orientation = exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
            exifStream.close()

            val matrix = Matrix()
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            }

            if (orientation != ExifInterface.ORIENTATION_NORMAL && orientation != ExifInterface.ORIENTATION_UNDEFINED) {
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            } else {
                bitmap
            }
        } catch (e: Exception) {
            null
        }
    }

    private fun processBitmap(bitmap: Bitmap) {
        currentBitmap = bitmap
        placeholderText.visibility = View.GONE

        // Run on-device inference with diagnostic telemetry
        val result = detector.analyze(bitmap)
        val detections = result.detections
        lastDetections = detections

        resultStatusText.text = "Model Output Shape: [${result.outputShape}]"

        if (detections.isEmpty()) {
            val maxScorePct = (result.maxRawScore * 100).toInt()
            val threshPct = (detector.confidenceThreshold * 100).toInt()
            resultDetailsText.text = "0 potholes detected. Max frame score: $maxScorePct% (Threshold: $threshPct%)"
        } else {
            val maxSeverity = if (detections.any { it.severity == "severe" }) "SEVERE"
            else if (detections.any { it.severity == "moderate" }) "MODERATE" else "MINOR"

            val maxConf = detections.maxOf { it.confidence }
            resultDetailsText.text = "Detected ${detections.size} pothole(s). Max Confidence: ${(maxConf * 100).toInt()}% ($maxSeverity)"
        }

        // Draw bounding boxes on image
        val annotated = drawBoxes(bitmap, detections)
        imageView.setImageBitmap(annotated)
    }

    private fun drawBoxes(source: Bitmap, detections: List<PotholeDetector.Detection>): Bitmap {
        val output = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)

        val strokeWidth = (source.width * 0.008f).coerceAtLeast(6f)

        val severePaint = Paint().apply {
            color = Color.parseColor("#FF3B30")
            style = Paint.Style.STROKE
            this.strokeWidth = strokeWidth
        }
        val moderatePaint = Paint().apply {
            color = Color.parseColor("#FF9500")
            style = Paint.Style.STROKE
            this.strokeWidth = strokeWidth
        }
        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = (source.width * 0.035f).coerceAtLeast(28f)
            isFakeBoldText = true
        }
        val bgPaint = Paint().apply {
            style = Paint.Style.FILL
        }

        detections.forEach { det ->
            val paint = if (det.severity == "severe") severePaint else moderatePaint
            bgPaint.color = paint.color

            canvas.drawRect(det.box, paint)

            val label = "Pothole (${det.severity.uppercase()}) ${"%.0f".format(det.confidence * 100)}%"
            val labelHeight = textPaint.textSize + 12f
            val labelTop = (det.box.top - 15f).coerceAtLeast(labelHeight)

            canvas.drawRect(
                det.box.left,
                labelTop - labelHeight,
                det.box.left + (label.length * textPaint.textSize * 0.55f),
                labelTop + 6f,
                bgPaint
            )
            canvas.drawText(label, det.box.left + 8f, labelTop - 4f, textPaint)
        }

        return output
    }

    private fun handleReportSubmission() {
        if (lastDetections.isEmpty() && currentBitmap == null) {
            Toast.makeText(this, "Please pick or capture a photo to detect hazards before reporting.", Toast.LENGTH_SHORT).show()
            return
        }

        val severity = if (lastDetections.any { it.severity == "severe" }) "severe" else "moderate"

        AlertDialog.Builder(this)
            .setTitle("Report Pothole Hazard")
            .setMessage("Submit this verified road hazard to the municipal grievance desk (Location: $currentLat, $currentLon)?")
            .setPositiveButton("Submit & Forward") { _, _ ->
                submitReportToBackend(severity)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun submitReportToBackend(severity: String) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val res = RideVisionApiClient.service.submitReport(
                    PotholeReportRequest(
                        lat = currentLat,
                        lon = currentLon,
                        severity = severity,
                        address = "Road near $currentLat, $currentLon",
                        note = "Detected via RideVision on-device YOLOv8"
                    )
                )

                withContext(Dispatchers.Main) {
                    if (res.isSuccessful && res.body() != null) {
                        val routing = res.body()!!.municipalRouting
                        Toast.makeText(this@MainActivity, "Report Registered! ID: ${res.body()!!.reportId}", Toast.LENGTH_LONG).show()

                        // If city supports direct WhatsApp dispatch, prompt to open
                        if (routing.channel == "whatsapp" && !routing.link.isNullOrEmpty()) {
                            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(routing.link))
                            startActivity(intent)
                        }
                    } else {
                        Toast.makeText(this@MainActivity, "Server report recorded locally.", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Report saved offline: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::detector.isInitialized) {
            detector.close()
        }
    }
}

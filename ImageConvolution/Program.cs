using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

// https://blog.demofox.org/2022/02/26/image-sharpening-convolution-kernels/
// https://en.wikipedia.org/wiki/Kernel_(image_processing)
// https://stackoverflow.com/questions/37095783/how-is-a-convolution-calculated-on-an-image-with-three-rgb-channels
// Image source https://www.nist.gov/image/03ts001peanutbsrmcsjpg
// Image source https://www.researchgate.net/profile/Shashikant-Ilager/publication/327134322/figure/fig5/AS:661945946492938@1534831620212/Edge-Detection-Sample-Input-and-Output-Images.png

namespace ImageConvolution
{
    struct Filter
    {
        public string Name { get; set; }
        public double[,] KernelHorizontal { get; set; }
        public double[,] KernelVertical { get; set; }
        public double Factor { get; set; }
        public int Bias { get; set; }
        public bool Grayscale { get; set; }
        public string BlurFilter { get; set; }

        public Filter(string name, double[,] kernel, double[,] kernelVertical = null, double factor = 1.0, int bias = 0, bool grayscale = false, string blurFilter = null)
        {
            Name = name;
            KernelHorizontal = kernel;
            KernelVertical = kernelVertical;
            Factor = factor;
            Bias = bias;
            Grayscale = grayscale;
            BlurFilter = blurFilter;
        }
    }

    class Program
    {
        private static readonly List<Filter> kernels =
        [
            new("Sharpen3x3", new double[,] {
                {  0, -1,  0 },
                { -1,  5, -1 },
                {  0, -1,  0 } }),
            new("Sharpen5x5", new double[,] {
                {  0,  0, -1,  0,  0 },
                {  0, -1, -1, -1,  0 },
                { -1, -1, 13, -1, -1 },
                {  0, -1, -1, -1,  0 },
                {  0,  0, -1,  0,  0 } }),
            new("BoxBlur3x3", new double[,] {
                { 1, 1, 1 },
                { 1, 1, 1 },
                { 1, 1, 1 } },
                factor: 1.0 / 9.0),
            new("BoxBlur9x9", new double[,] {
                { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
                { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
                { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
                { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
                { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
                { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
                { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
                { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
                { 1, 1, 1, 1, 1, 1, 1, 1, 1 } },
                factor: 1.0 / 81.0),
            new("GaussianBlur3x3", new double[,] {
                { 1, 2, 1 },
                { 2, 4, 2 },
                { 1, 2, 1 } },
                factor: 1.0 / 16.0),
            new("GaussianBlur5x5", new double[,] {
                { 2,  4,  5,  4,  2 },
                { 4,  9, 12,  9,  4 },
                { 5, 12, 15, 12,  5 },
                { 4,  9, 12,  9,  4 },
                { 2,  4,  5,  4,  2 } },
                factor: 1.0 / 159.0),
            new("UnsharpMasking_BoxBlur3x3", new double[,] {
                { -1, -1, -1 },
                { -1, 17, -1 },
                { -1, -1, -1 } },
                factor: 1.0 / 9.0),
            new("UnsharpMasking_GaussianBlur3x3", new double[,] {
                { -0.0023, -0.0432, -0.0023 },
                { -0.0432,   1.182, -0.0432 },
                { -0.0023, -0.0432, -0.0023 } }),
            new("Laplacian3x3_Grayscale", new double[,] {
                {  0, -1,  0 },
                { -1,  4, -1 },
                {  0, -1,  0 } },
                grayscale: true),
            new("Laplacian3x3_Type2_Grayscale", new double[,] {
                { -1, -1, -1 },
                { -1,  8, -1 },
                { -1, -1, -1 } },
                grayscale: true),
            new("Laplacian5x5", new double[,] {
                { -1, -1, -1, -1, -1, },
                { -1, -1, -1, -1, -1, },
                { -1, -1, 24, -1, -1, },
                { -1, -1, -1, -1, -1, },
                { -1, -1, -1, -1, -1  } },
                grayscale: false),
            new("Laplacian5x5_GaussianBlur3x3", new double[,] {
                { -1, -1, -1, -1, -1, },
                { -1, -1, -1, -1, -1, },
                { -1, -1, 24, -1, -1, },
                { -1, -1, -1, -1, -1, },
                { -1, -1, -1, -1, -1  } },
                grayscale: false, blurFilter: "GaussianBlur3x3"),
            new("Laplacian5x5_Grayscale", new double[,] {
                { -1, -1, -1, -1, -1, },
                { -1, -1, -1, -1, -1, },
                { -1, -1, 24, -1, -1, },
                { -1, -1, -1, -1, -1, },
                { -1, -1, -1, -1, -1  } },
                grayscale: true),
            new("Laplacian5x5_GaussianBlur3x3_Grayscale", new double[,] {
                { -1, -1, -1, -1, -1, },
                { -1, -1, -1, -1, -1, },
                { -1, -1, 24, -1, -1, },
                { -1, -1, -1, -1, -1, },
                { -1, -1, -1, -1, -1  } },
                grayscale: true, blurFilter: "GaussianBlur3x3"),
            new("Laplacian5x5_GaussianBlur5x5_Grayscale", new double[,] {
                { -1, -1, -1, -1, -1, },
                { -1, -1, -1, -1, -1, },
                { -1, -1, 24, -1, -1, },
                { -1, -1, -1, -1, -1, },
                { -1, -1, -1, -1, -1  } },
                grayscale: true, blurFilter: "GaussianBlur5x5"),
            new("Sobel3x3_Grayscale", new double[,]  {
                { -1, 0, 1, },
                { -2, 0, 2, },
                { -1, 0, 1, } },
                kernelVertical: new double[,] {
                    {  1,  2,  1, },
                    {  0,  0,  0, },
                    { -1, -2, -1, } },
                grayscale: true),
            new("Sobel3x3", new double[,] {
                { -1, 0, 1, },
                { -2, 0, 2, },
                { -1, 0, 1, } },
                kernelVertical: new double[,] {
                    {  1,  2,  1, },
                    {  0,  0,  0, },
                    { -1, -2, -1, } },
                grayscale: false),
            new("Prewitt3x3_Grayscale", new double[,]  {
                { -1, 0, 1, },
                { -1, 0, 1, },
                { -1, 0, 1, } },
                kernelVertical: new double[,] {
                    {  1,  1,  1, },
                    {  0,  0,  0, },
                    { -1, -1, -1, } },
                grayscale: true),
            new("Prewitt3x3", new double[,]  {
                { -1, 0, 1, },
                { -1, 0, 1, },
                { -1, 0, 1, } },
                kernelVertical: new double[,] {
                    {  1,  1,  1, },
                    {  0,  0,  0, },
                    { -1, -1, -1, } },
                grayscale: false),
            new("Kirsch3x3_Grayscale", new double[,]  {
                {  5,   5,   5, },
                { -3,   0,  -3, },
                { -3,  -3,  -3, } },
                kernelVertical: new double[,] {
                    { 5, -3, -3, },
                    { 5,  0, -3, },
                    { 5, -3, -3, } },
                grayscale: true),
            new("Kirsch3x3", new double[,]  {
                {  5,   5,   5, },
                { -3,   0,  -3, },
                { -3,  -3,  -3, } },
                kernelVertical: new double[,] {
                    { 5, -3, -3, },
                    { 5,  0, -3, },
                    { 5, -3, -3, } },
                grayscale: false),
        ];

        private static readonly PixelFormat pixelFormat = PixelFormat.Format32bppArgb;
        private static readonly int bytesPerPixel = 4;

        public static void Main()
        {
            string curDir = Path.GetDirectoryName(Assembly.GetEntryAssembly().Location);
            string[] jpgFiles = Directory.GetFiles(curDir, "*.jpeg");
            string[] tiffFiles = Directory.GetFiles(curDir, "*.tiff");
            foreach (string file in jpgFiles.Union(tiffFiles))
            {
                File.Delete(file);
            }

            string path = "0_input.jpg";
            int fileCount = 1;

            using StreamReader streamReader = new(path);
            Bitmap bmp = (Bitmap)Image.FromStream(streamReader.BaseStream);
            streamReader.Close();

            Console.WriteLine($"Loaded image {path}");
            Console.WriteLine();

            byte[] imageBytes = BitmapToByteArray(bmp, pixelFormat, bytesPerPixel);

            foreach (var kernel in kernels)
            {
                byte[] input = new byte[imageBytes.Length];
                Array.Copy(imageBytes, 0, input, 0, imageBytes.Length);

                string message = $"Applying {kernel.Name}";
                if (kernel.Grayscale)
                {
                    message += " + Grayscale";
                }
                if (!string.IsNullOrEmpty(kernel.BlurFilter))
                {
                    message += $" + Blur filter: {kernel.BlurFilter}";
                }
                Console.WriteLine($"{message}...");

                if (kernel.Grayscale)
                {
                    ConvertToGrayscale(ref input, bytesPerPixel);
                    //SaveImage(input, bmp.Width, bmp.Height, pixelFormat, "1_grayscale", fileCount);
                }

                if (!string.IsNullOrEmpty(kernel.BlurFilter))
                {
                    Filter blurKernel = kernels.Find(x => x.Name == kernel.BlurFilter);
                    input = Convolve(input, blurKernel, bmp.Width, bmp.Height, bytesPerPixel);
                    //SaveImage(input, bmp.Width, bmp.Height, pixelFormat, "2_blur", fileCount);
                }

                byte[] output = Convolve(input, kernel, bmp.Width, bmp.Height, bytesPerPixel);

                SaveImage(output, bmp.Width, bmp.Height, pixelFormat, kernel.Name, fileCount++);
                Console.WriteLine();
            }
        }

        public static byte[] Convolve(
            byte[] imageBytes,
            Filter filter,
            int width,
            int height,
            int bytesPerPixel)
        {
            byte[] output = new byte[imageBytes.Length];
            int stride = width * bytesPerPixel;

            int filterHeight = filter.KernelHorizontal.GetLength(0);
            int filterWidth = filter.KernelHorizontal.GetLength(1);
            int filterOffset = (filterWidth - 1) / 2;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    double bx, gx, rx;
                    double by, gy, ry;
                    double btotal, gtotal, rtotal;
                    bx = gx = rx = by = gy = ry = 0;

                    for (int f_y = 0; f_y < filterHeight; f_y++)
                    {
                        for (int f_x = 0; f_x < filterWidth; f_x++)
                        {
                            int row = y + (f_y - filterOffset);
                            int col = x + (f_x - filterOffset);

                            // Edge handling: Extend
                            row = Math.Min(Math.Max(row, 0), height - 1);
                            col = Math.Min(Math.Max(col, 0), width - 1);

                            int calcOffset = (row * stride) + (col * bytesPerPixel);

                            double fvx = filter.KernelHorizontal[f_y, f_x];
                            bx += imageBytes[calcOffset] * fvx;
                            gx += imageBytes[calcOffset + 1] * fvx;
                            rx += imageBytes[calcOffset + 2] * fvx;

                            if (filter.KernelVertical != null)
                            {
                                double fvy = filter.KernelVertical[f_y, f_x];
                                by += imageBytes[calcOffset] * fvy;
                                gy += imageBytes[calcOffset + 1] * fvy;
                                ry += imageBytes[calcOffset + 2] * fvy;
                            }
                        }
                    }

                    bx = (filter.Factor * bx) + filter.Bias;
                    gx = (filter.Factor * gx) + filter.Bias;
                    rx = (filter.Factor * rx) + filter.Bias;

                    if (filter.KernelVertical != null)
                    {
                        by = (filter.Factor * by) + filter.Bias;
                        gy = (filter.Factor * gy) + filter.Bias;
                        ry = (filter.Factor * ry) + filter.Bias;

                        btotal = Math.Sqrt((bx * bx) + (by * by));
                        gtotal = Math.Sqrt((gx * gx) + (gy * gy));
                        rtotal = Math.Sqrt((rx * rx) + (ry * ry));
                    }
                    else
                    {
                        btotal = bx;
                        gtotal = gx;
                        rtotal = rx;
                    }

                    btotal = Math.Min(255, Math.Max(0, btotal));
                    gtotal = Math.Min(255, Math.Max(0, gtotal));
                    rtotal = Math.Min(255, Math.Max(0, rtotal));

                    int sourceOffset = (y * stride) + (x * bytesPerPixel);

                    output[sourceOffset] = (byte)btotal;
                    output[sourceOffset + 1] = (byte)gtotal;
                    output[sourceOffset + 2] = (byte)rtotal;
                    output[sourceOffset + 3] = 255;
                }
            }

            return output;
        }

        private static byte[] BitmapToByteArray(Bitmap bmp, PixelFormat pixelFormat, int bytesPerPixel)
        {
            BitmapData bmpdata = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadOnly, pixelFormat);
            byte[] pixelBuffer = new byte[bmpdata.Stride * bmp.Height];
            Marshal.Copy(bmpdata.Scan0, pixelBuffer, 0, pixelBuffer.Length);
            bmp.UnlockBits(bmpdata);
            return pixelBuffer;
        }

        private static void SaveImage(byte[] imageBytes, int width, int height, PixelFormat format, string filename, int fileCount)
        {
            Bitmap output_bmp = ByteArrayToBitmap(imageBytes, width, height, format);
            string fullFileName = $"{fileCount}_{filename}.jpeg";
            output_bmp.Save(fullFileName);
            Console.WriteLine($"Saved image file {fullFileName}");
        }

        private static Bitmap ByteArrayToBitmap(byte[] bytes, int width, int height, PixelFormat pixelFormat)
        {
            Bitmap bmp = new(width, height, pixelFormat);
            BitmapData bmpData = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.WriteOnly, bmp.PixelFormat);
            Marshal.Copy(bytes, 0, bmpData.Scan0, bytes.Length);
            bmp.UnlockBits(bmpData);
            return bmp;
        }

        private static void ConvertToGrayscale(ref byte[] pixelBuffer, int bytesPerPixel)
        {
            for (int i = 0; i < pixelBuffer.Length; i += bytesPerPixel)
            {
                float luminosity = pixelBuffer[i] * 0.11f;
                luminosity += pixelBuffer[i + 1] * 0.59f;
                luminosity += pixelBuffer[i + 2] * 0.3f;

                pixelBuffer[i] = (byte)luminosity;
                pixelBuffer[i + 1] = (byte)luminosity;
                pixelBuffer[i + 2] = (byte)luminosity;
                if (bytesPerPixel == 4)
                {
                    pixelBuffer[i + 3] = 255;
                }
            }
        }
    }
}

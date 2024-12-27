using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;

// https://blog.demofox.org/2022/02/26/image-sharpening-convolution-kernels/
// https://en.wikipedia.org/wiki/Kernel_(image_processing)
// https://stackoverflow.com/questions/37095783/how-is-a-convolution-calculated-on-an-image-with-three-rgb-channels
// Image source https://www.nist.gov/image/03ts001peanutbsrmcsjpg

namespace ImageConvolution
{
    struct Kernel(string name, double[,] matrix, bool grayscale = false, bool blur = false)
    {
        public string Name { get; set; } = name;
        public double[,] Matrix { get; set; } = matrix;
        public bool Grayscale { get; set; } = grayscale;
        public bool Blur { get; set; } = blur;
    }

    class Program
    {
        private static readonly List<Kernel> kernels =
        [
            new("Sharpen", new double[,] {
                { 0, -1, 0 },
                { -1, 5, -1 },
                { 0, -1, 0 } }),
            new("Box_blur", new double[,] {
                { 0.1111, 0.1111, 0.1111 },
                { 0.1111, 0.1111, 0.1111 },
                { 0.1111, 0.1111, 0.1111 } }),
            new("Gaussian_blur_3_3", new double[,] {
                { 0.0625, 0.125, 0.0625 },
                { 0.125,   0.25, 0.125 },
                { 0.0625, 0.125, 0.0625 } }),
            new("Unsharp_masking_box_blur", new double[,] {
                { -0.1111, -0.1111, -0.1111 },
                { -0.1111,  1.8888, -0.1111 },
                { -0.1111, -0.1111, -0.1111 } }),
            new("Unsharp_masking_gaussian_blur", new double[,] {
                { -0.0023, -0.0432, -0.0023 },
                { -0.0432,   1.182, -0.0432 },
                { -0.0023, -0.0432, -0.0023 } }),
            new("Laplacian_1", new double[,] {
                {  0, -1,  0 },
                { -1,  4, -1 },
                {  0, -1,  0 } }, grayscale: true, blur: false),
            new("Laplacian_2", new double[,] {
                { -1, -1, -1 },
                { -1,  8, -1 },
                { -1, -1, -1 } }, grayscale: true, blur: false),
        ];

        private static readonly PixelFormat pixelFormat = PixelFormat.Format24bppRgb;
        private static readonly int bytesPerPixel = 3;

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
            byte[] output = new byte[imageBytes.Length];

            int stride = bmp.Width * bytesPerPixel;

            foreach (var kernel in kernels)
            {
                Console.WriteLine($"Applying {kernel.Name}...");

                if (kernel.Grayscale)
                {
                    ConvertToGrayscale(ref imageBytes, bytesPerPixel);
                    SaveImage(imageBytes, bmp.Width, bmp.Height, pixelFormat, "1_grayscale", fileCount);
                }

                if (kernel.Blur)
                {
                    Kernel blurKernel = kernels.Find(x => x.Name == "Gaussian_blur_3_3");
                    Convolve(imageBytes, blurKernel, imageBytes, bmp.Width, bmp.Height, bytesPerPixel, stride);
                    SaveImage(imageBytes, bmp.Width, bmp.Height, pixelFormat, "2_blur", fileCount);
                }

                Convolve(imageBytes, kernel, output, bmp.Width, bmp.Height, bytesPerPixel, stride);

                SaveImage(output, bmp.Width, bmp.Height, pixelFormat, kernel.Name, fileCount++);
            }
        }

        public static void Convolve(byte[] imageBytes, Kernel kernel, byte[] output, int width, int height, int bytesPerPixel, int stride)
        {
            for (int x = 0; x < height; x++)
            {
                for (int y = 0; y < width; y++)
                {
                    double b = 0.0;
                    double g = 0.0;
                    double r = 0.0;

                    for (int f_y = 0; f_y < kernel.Matrix.GetLength(0); f_y++)
                    {
                        for (int f_x = 0; f_x < kernel.Matrix.GetLength(1); f_x++)
                        {
                            int row = x + (f_y - 1);
                            int col = y + (f_x - 1);

                            // Edge handling: Extend
                            row = Math.Min(Math.Max(row, 0), height - 1);
                            col = Math.Min(Math.Max(col, 0), width - 1);

                            int calcOffset = (row * stride) + (col * bytesPerPixel);
                            double filterValue = kernel.Matrix[f_y, f_x];

                            b += imageBytes[calcOffset] * filterValue;
                            g += imageBytes[calcOffset + 1] * filterValue;
                            r += imageBytes[calcOffset + 2] * filterValue;
                        }
                    }

                    b = Math.Min(255, Math.Max(0, b));
                    g = Math.Min(255, Math.Max(0, g));
                    r = Math.Min(255, Math.Max(0, r));

                    int sourceOffset = (x * stride) + (y * bytesPerPixel);

                    output[sourceOffset] = (byte)b;
                    output[sourceOffset + 1] = (byte)g;
                    output[sourceOffset + 2] = (byte)r;
                    if (bytesPerPixel == 4)
                    {
                        output[sourceOffset + 3] = 255;
                    }
                }
            }
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

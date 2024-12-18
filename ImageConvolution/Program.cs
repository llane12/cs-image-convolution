using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

// https://blog.demofox.org/2022/02/26/image-sharpening-convolution-kernels/
// https://en.wikipedia.org/wiki/Kernel_(image_processing)
// https://stackoverflow.com/questions/37095783/how-is-a-convolution-calculated-on-an-image-with-three-rgb-channels
// Image source https://www.nist.gov/image/03ts001peanutbsrmcsjpg

string path = "03ts001_peanutbsrm_cs-cropped.jpg";
Image image = Image.FromFile(path);

Bitmap bmp = new Bitmap(image);

PixelFormat format = PixelFormat.Format24bppRgb;
const int bytesPerPixel = 3;
byte[] bytes = BitmapToByteArray(bmp, format, bytesPerPixel);

Console.WriteLine($"Loaded image {path} using format {format} into byte array[{bytes.Length}]");

List<Tuple<string, double[,]>> kernels =
[
    new("Sharpen", new double[,] {
        { 0, -1, 0 },
        { -1, 5, -1 },
        { 0, -1, 0 } }),
    new("Box_blur", new double[,] {
        { 0.1111, 0.1111, 0.1111 },
        { 0.1111, 0.1111, 0.1111 },
        { 0.1111, 0.1111, 0.1111 } }),
    new("Unsharp_masking_box_blur", new double[,] {
        { -0.1111, -0.1111, -0.1111 },
        { -0.1111, 1.8888, -0.1111 },
        { -0.1111, -0.1111, -0.1111 } }),
    new("Unsharp_masking_gaussian_blur", new double[,] {
        { -0.0023, -0.0432, -0.0023 },
        { -0.0432,   1.182, -0.0432 },
        { -0.0023, -0.0432, -0.0023 } }),
];

byte[] output = new byte[bytes.Length];

int stride = image.Width * bytesPerPixel;

foreach (var kernel_type in kernels)
{
    string name = kernel_type.Item1;
    double[,] kernel = kernel_type.Item2;

    Console.WriteLine($"Applying {name}...");

    for (int i = 0; i < bytes.Length; i += bytesPerPixel)
    {
        int row = i / stride;
        int col = (i - (row * stride)) / bytesPerPixel;

        for (int j = 0; j < bytesPerPixel; j++) // B G R
        {
            double accumulator = 0;

            for (int y = 0; y < kernel.GetLength(0); y += 1)
            {
                for (int x = 0; x < kernel.GetLength(1); x += 1)
                {
                    int r = row + (y - 1);
                    int c = col + (x - 1);

                    // Edge handling: Extend
                    r = Math.Min(Math.Max(r, 0), image.Height - 1);
                    c = Math.Min(Math.Max(c, 0), image.Width - 1);

                    int offset = (r * stride) + (c * bytesPerPixel);
                    byte value = bytes[offset + j];

                    accumulator += value * kernel[y, x];
                }
            }

            byte acc = (byte)accumulator;
            output[i + j] = acc;
        }
    }

    Bitmap output_bmp = ByteArrayToBitmap(output, image.Width, image.Height, format);
    output_bmp.Save($"{name}.jpg");
    Console.WriteLine($"Saved image file {name}.jpg");
}

byte[] BitmapToByteArray(Bitmap bmp, PixelFormat pixelFormat, int bytesPerPixel)
{
    BitmapData bmpdata = null;

    try
    {
        bmpdata = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadOnly, pixelFormat);
        // For some reason bmpdata.Stride is not correct, so pass in the expected number of bytes per pixel and calculate the stride
        int numbytes = bmp.Width * bmp.Height * bytesPerPixel;
        byte[] bytedata = new byte[numbytes];
        IntPtr ptr = bmpdata.Scan0;
        Marshal.Copy(ptr, bytedata, 0, numbytes);
        return bytedata;
    }
    finally
    {
        if (bmpdata != null)
            bmp.UnlockBits(bmpdata);
    }
}

Bitmap ByteArrayToBitmap(byte[] bytes, int width, int height, PixelFormat pixelFormat)
{
    Bitmap bmp = new Bitmap(width, height, pixelFormat);
    BitmapData bmpData = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.WriteOnly, bmp.PixelFormat);
    Marshal.Copy(bytes, 0, bmpData.Scan0, bytes.Length);
    bmp.UnlockBits(bmpData);
    return bmp;
}

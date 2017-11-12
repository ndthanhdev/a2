using System;
using HtmlAgilityPack;


namespace runtime
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            var url = $"https://vnexpress.net/tin-tuc/khoa-hoc/hoi-dap/page/{56}.html";
            var web = new HtmlWeb();
            web.CaptureRedirect = false;
            var doc = web.Load(url);

        }
    }
}
